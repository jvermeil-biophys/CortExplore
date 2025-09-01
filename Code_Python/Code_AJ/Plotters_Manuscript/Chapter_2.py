# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:22:32 2024

@author: anumi
"""

# %% > Imports and constants



#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st

import statsmodels.api as sm


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan')

import os
import sys
import time
import random
import warnings
import itertools
import matplotlib
import distinctipy

from copy import copy
from cycler import cycler
from datetime import date
import matplotlib.lines as lines
import PlottingFunctions_AJ as pf
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, wilcoxon
# from statannotations.Annotator import Annotator

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun

import TrackAnalyser as taka0
import TrackAnalyser_V3_Manuscript_AJ as taka


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
fontColour = '#ffffff'

plt.rcParams.update({
    'font.family': 'Arial',   # Choose your font
})

SCALE_px_cm = 2.60


# %%%% Default settings for mechanics analysis

#### HOW TO USE    
# Example : I don't want to do a whole curve Chawick Fit anymore, but I want a Dimitriadis one.
# I can create a dict fitSettings = {'doChadwickFit' : False, 'doDimitriadisFit' : True}
# And pass it as an argument in computeGlobalTable_meca

#### DETAILS
# See ufun.updateDefaultSettingsDict(settingsDict, defaultSettingsDict)
# And the 'Settings' flag in analyseTimeSeries_meca() 

#### AS REFERENCE ONLY, here is a copy of the default settings.

#### 1. For Fits
DEFAULT_stressCenters = [ii for ii in range(100, 1550, 50)]
DEFAULT_stressHalfWidths = [50, 75, 100]

DEFAULT_strainCenters = [ii/10000 for ii in range(125, 3750, 125)]
DEFAULT_strainHalfWidths = [0.0125, 0.025, 0.05]

DEFAULT_fitSettings = {# H0
                       'methods_H0':['Chadwick', 'Dimitriadis'],
                       'zones_H0':['%f_10', '%f_20'],
                       'method_bestH0':'Chadwick',
                       'zone_bestH0':'%f_10',
                       # Global fits
                       'doVWCFit' : True,
                       'VWCFitMethods' : ['Full'],
                       'doDimitriadisFit' : False,
                       'DimitriadisFitMethods' : ['Full'],
                       'doChadwickFit' : True,
                       'ChadwickFitMethods' : ['Full', 'f_<_400', 'f_in_400_800'],
                       'doDimitriadisFit' : False,
                       'DimitriadisFitMethods' : ['Full'],
                       # Local fits
                       'doStressRegionFits' : False,
                       'doStressGaussianFits' : False,
                       'centers_StressFits' : DEFAULT_stressCenters,
                       'halfWidths_StressFits' : DEFAULT_stressHalfWidths,
                       'doNPointsFits' : False,
                       'nbPtsFit' : 13,
                       'overlapFit' : 3,
                       # NEW - Numi
                       'doLogFits' : False,
                       'nbPtsFitLog' : 10,
                       'overlapFitLog' : 5,
                       # NEW - Jojo
                       'doStrainGaussianFits' : False,
                       'centers_StrainFits' : DEFAULT_strainCenters,
                       'halfWidths_StrainFits' : DEFAULT_strainHalfWidths,
                       # TEST - Jojo
                       'do3partsFits' : False,
                       }

#### 2. For Validation

DEFAULT_crit_nbPts = 8 # sup or equal to
DEFAULT_crit_R2 = 0.6 # sup or equal to
DEFAULT_crit_Chi2 = 1 # inf or equal to
DEFAULT_str_crit = 'nbPts>{:.0f} - R2>{:.2f} - Chi2<{:.1f}'.format(DEFAULT_crit_nbPts, 
                                                                   DEFAULT_crit_R2, 
                                                                   DEFAULT_crit_Chi2)

DEFAULT_fitValidationSettings = {'crit_nbPts': DEFAULT_crit_nbPts, 
                                 'crit_R2': DEFAULT_crit_R2, 
                                 'crit_Chi2': DEFAULT_crit_Chi2,
                                 'str': DEFAULT_str_crit}

#### 3. For Plots
DEFAULT_plot_stressCenters = [ii for ii in range(100, 1550, 50)]
DEFAULT_plot_stressHalfWidth = 75

DEFAULT_plot_strainCenters = [ii/10000 for ii in range(125, 3750, 125)]
DEFAULT_plot_strainHalfWidth = 0.0125

DEFAULT_plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'F(H)_VWC':True,
                        'S(e)_stressRegion':True,
                        'K(S)_stressRegion':True,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_Log':True, # NEW - Numi
                        'K(S)_Log':True, # NEW - Numi
                        'S(e)_strainGaussian':True, # NEW - Jojo
                        'K(S)_strainGaussian':True, # NEW - Jojo
                        'Plot_Ratio':True, # NEW
                        # Fits plotting parameters
                        # Stress
                        'plotStressCenters':DEFAULT_plot_stressCenters,
                        'plotStressHW':DEFAULT_plot_stressHalfWidth,
                        # Strain
                        'plotStrainCenters':DEFAULT_plot_strainCenters,
                        'plotStrainHW':DEFAULT_plot_strainHalfWidth,
                        # Points
                        'plotPoints':str(DEFAULT_fitSettings['nbPtsFit']) \
                                     + '_' + str(DEFAULT_fitSettings['overlapFit']),
                        'plotLog':str(DEFAULT_fitSettings['nbPtsFitLog']) \
                                     + '_' + str(DEFAULT_fitSettings['overlapFitLog']),
                        }
    

#%% Mechanics


plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick', 'VWC', 'Dimitriadis'],
                'zones_H0':['%f_10', '%f_15', '%f_100'],
                'method_bestH0':'VWC', 
                'zone_bestH0':'%f_100',
                'doVWCFit' : True,
                'VWCFitMethods' : ['Full'],
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full', 'f_<_400'],
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : False,
                'nbPtsFit' : 33,
                'overlapFit' : 21,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : False,
                }

plot_stressCenters = [ii for ii in range(100, 4000, 100)]
plot_stressHalfWidth = 75

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'F(H)_VWC':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':False,
                        'K(S)_stressGaussian':False,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':False,
                        'K(S)_nPoints':False,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        'Plot_Ratio':False
                        }


# fitSettings = {# H0
#                 'methods_H0':['Chadwick', 'VWC', 'Dimitriadis'],
#                 'zones_H0':['%f_10', '%f_15', '%f_100'],
#                 'method_bestH0':'Chadwick', 
#                 'zone_bestH0':'%f_15',
#                 'doVWCFit' : True,
#                 'VWCFitMethods' : ['Full'],
#                 'doChadwickFit' : True,
#                 'ChadwickFitMethods' : ['Full', 'f_<_400'],
#                 'doStressRegionFits' : False,
#                 'doStressGaussianFits' : False,
#                 'centers_StressFits' : plot_stressCenters,
#                 'halfWidths_StressFits' : stressHalfWidths,
#                 'doNPointsFits' : False,
#                 'nbPtsFit' : 33,
#                 'overlapFit' : 21,
#                 # NEW - Numi
#                 'doLogFits' : False,
#                 # NEW - Jojo
#                 'doStrainGaussianFits' : False,
#                 }


# Task = '24-05-29 & 24-09-05_M0 & 24-09-05_M1 & 24-09-05_M3 & 24-09-05_M4 & 24-02-21 & 24-09-24 & 24-09-12'
# Task = '23-02-16 & 23-02-23 & 23-03-08 & 23-03-09 & 23-03-16 & 23-03-17 & 23-03-24 & 22-12-07_M4 & 23-02-02_M1 & 23-01-23_M1 & 23-04-25 & 23-05-10 & 23-05-23_M1 & 23-07-07_M2 & 23-07-12_M1'
Task = '23-07-12_M1_P1_C4'

        
plt.style.use(())
fitsSubDir ='VWC-Chadwick_Chapter-2_23-07-12_M1_P1_C4'


sns.set_theme()

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = True, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'



#%% Calling data - Only C-OptoRhoA, testing effects of 5mT ad 15mT on mechanics

"""
Task = '23-06-28 & 23-07-07'

'VWC-Chadwick_Chapter-2_C-OptoRhoA_5mTv15mT_25-04-14'
        
"""

filename = 'VWC-Chadwick_Chapter-2_C-OptoRhoA_5mTv15mT_25-04-14'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
data = pf.createDataTable(GlobalTable)

fitsSubDir = 'VWC-Chadwick_Chapter-2_C-OptoRhoA_5mTv15mT_25-04-14'

#%%% Create dataframe for plotting

styleDict =  {#5.0:{'color': '#20a39a','marker':'o', 'label': '5mT'},
               15.0:{'color': '#2478b7','marker':'o', 'label': '15mT'},
               }

celltypes = ['optoRhoA']
magField = [15.0]#, 15.0]
drugs = ['none']

# # dates = ['23-06-28', '23-07-07' ]
# dates = ['23-07-07'] #, '23-07-07' ]

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 10),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['compression duration'] == '1.5s'),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['drug'].apply(lambda x : x in drugs)),
            # (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
df_h0 = taka.getMatchingFits(df, fitsSubDir = fitsSubDir, fitType = 'H0')
h0_method = ['Chadwick', '%f_15']
df_h0 = df_h0[(df_h0['method'] == h0_method[0]) & (df_h0['zone'] == h0_method[1])]
df = pd.merge(df, df_h0[['cellID', 'compNum', 'H0', 'nbPts']], on = ['cellID', 'compNum'], how = 'left')
df = df.rename(columns={'H0':'{:}_{:}_H0'.format(h0_method[0], h0_method[1])})

df = pf.NLIcorr(df)

condCol, condCat = 'normal field', magField

styleDf = pd.DataFrame(styleDict)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleDict)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]


#%%% Chadwick %f_10


y = 'surroundingThickness' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'median'),
                  'order' : condCat,
                  's':6,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
# plt.yscale('log')
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

# y_labels = np.asarray([100 ,250, 500, 1000, 1500])
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

y_labels = np.asarray([0,250, 500, 750, 1000, 1250, 1500])
# y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_labels, labels =y_labels,**plotTicks)

x_labels = ['5mT\n~50pN', '15mT\n~500pN']
x_ticks = [0,1]
ax.set_xticks(x_ticks, labels =x_labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness \n @ 15% Max Force (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '5mTv15mT_{:}.pdf').format(str(y)), dpi = 200)

#%%% Elasticity < 400pN


y = 'E_f_<_400_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':6,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }




fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
y_labels = np.asarray([500, 1000, 2500, 5000, 10000, 20000, 30000])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =(y_labels)/1000,**plotTicks)

x_labels = ['5mT\n~50pN', '15mT\n~500pN']
x_ticks = [0,1]
ax.set_xticks(x_ticks, labels =x_labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Elasticity < 400pN (kPa)', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + '5mTv15mT_{:}.pdf').format('E_f_400_log'), dpi = 200)

#%%% Rain plots, with distributions

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(15/SCALE_px_cm, 15/SCALE_px_cm))

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-3, 3)
plt.ylabel('', **plotChars)
plt.xlabel('', **plotChars)

plt.tight_layout()
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)

plt.savefig((dirToSave + '5mT-15mT_C-Opto_AvgNLRrainplot.pdf').format( str(condCat)), dpi = 100)
plt.show()

#%%% Fluctuations vs Thickness

fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm))

plottingParams = {
                  'x' :avgDf[('Chadwick_%f_15_H0' , 'mean')],
                  'y':avgDf[('ctFieldFluctuAmpli' , 'first')],
                  's':30
                    }

params, results = ufun.fitLineHuber(avgDf[('Chadwick_%f_15_H0' , 'mean')], avgDf[('ctFieldFluctuAmpli' , 'first')])
pval = results.pvalues[1]
a, k = params[1], params[0]
x = np.linspace(100, 1200, 10)
y = a*x + k

eqnText = ''
eqnText += " Fit y = m * x + c\n".format(a, k)
eqnText += " y = {:.1e} * x + {:.1f}\n".format(a, k)
eqnText += " p-val = {:.1e}".format(pval)
# label = eqnText, 
ax = sns.scatterplot( # color = styleDict[magField[0]]['color'],
, **plottingParams)
plt.plot(x, y, color = styleDict[magField[0]]['color'])

plt.legend(fontsize=12)
# plt.xticks(**plotTicks)
# plt.yticks(**plotTicks)
ax.set_ylim(0, 1000)
plt.show()
plt.savefig((dirToSave + 'Fluctuations_bestH0_{:}.pdf').format(str(magField), dpi = 200))
#%%% Cell-dependent NLR Variability

plotter = df[df['date'] =='23-07-07']

plotter = plotter.groupby('cellID', \
                          group_keys=False).apply(pf.NLR_normalize_by_first_six,
                                                  'NLI_mod', clip = (0, 6),
                                                  col = 'compNum')


plotter2 = avgDf[('cellID', 'first')]
colorblind_type = "Deuteranomaly"

unique_cellIDs = plotter['cellID'].unique()
Ncells = len(unique_cellIDs)
palette = distinctipy.get_colors(Ncells, colorblind_type=colorblind_type)

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))
cellID_to_color = {cell: palette[i] for i, cell in enumerate(unique_cellIDs)}

plottingParams = {'data':plotter, 
                  'x' : ('cellID'), 
                  'y' : ('NLI_mod_norm'),
                  'hue':'cellID',
                  'palette':cellID_to_color,
                  's':4
                 }

sns.swarmplot(**plottingParams, ax = ax)
for collection in ax.collections:
    collection.set_edgecolor('k')
    collection.set_linewidth(0.5) 


avg_std = plotter['NLI_mod_std'].mean()

for _, row in plotter.iterrows():
    plt.errorbar(x=row['cellID'], y=row['NLI_mod_ref'], yerr=row['NLI_mod_std'],
                 fmt="none", color=cellID_to_color[row['cellID']],
                 capsize=5, alpha=0.4, lw = 0.3,  capthick=1)
    
    
text = 'Average STD of Cells = {:.2f}'.format(avg_var)

plt.text(-0.5, 3, text, fontsize=12)
x_labels = ['C'+str(i+1) for i in range(Ncells)]
ax.set_xticks(ax.get_xticks(), labels =x_labels, rotation = 90, **plotTicks)
plt.ylim(-3, 3.5)

plt.savefig((dirToSave + '{:}-COptoRhoA_NLI_StandardDev.pdf').format(condCat), dpi = 200)

#%%% Cell-dependent NLR evolution

plotter = avgDf[avgDf[('date', 'first')] == '23-07-07']

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))


plottingParams = {'data':plotter, 
                  'x' : ('date', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'hue':('cellID','first'),
                  'palette':cellID_to_color,
                  's':10
                 }

sns.swarmplot(**plottingParams, ax = ax)
plt.ylim(-3, 3.5)

std_pop = plotter[('NLI_mod', 'mean')].std()
text = 'STD of Population = {:.2f}'.format(std_pop)
plt.text(0, 3, text, fontsize=12)

plt.savefig((dirToSave + '{:}-COptoRhoA_NLI_StdDevofMean.pdf').format(condCat), dpi = 200)

#%%% Plotnine jitter plots
#%%%% NLI
measure = 'NLI_mod'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, pointSize = 2.5,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1, 2))

plt.xticks([1,2,3], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + '5mT-15mT_C-Opto.pdf'))

#%%%% H0

measure = 'Chadwick_%f_10_H0'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, logScale = True,
                     pairs = pairs, stat = stat, palette = palette_cond, pointSize = 2.5,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)

plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + '5mT-15mT_C-Opto.pdf'))


#%%%% E_effective

measure = 'E_f_<_400_log'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure,  pointSize = 2.5,
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)

plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Elasticity < 400pN (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave  + 'E_f_400pN_5mT-15mT_C-Opto.pdf'))

#%% Calling data - All Data, focussing on 3T3 WT and C-OptoRhoA (no light)

"""
'VWC-Chadwick_Chapter-2_25-03-27' 
Task = '23-02-16 & 23-02-23 & 23-03-08 & 23-03-09 & 23-03-16 & 23-03-17 & 23-03-24 &'\
        '22-12-07 & 23-02-02 & 23-01-23 & 23-03-24 & 23-03-28 & 23-04-19 & 23-04-25 &'\
        '23-05-10 & 23-05-23 & 23-06-28 & 23-07-07 & 23-07-12 &'\
        '24-12-14 & 24-12-20 & 25-01-14 & 25-01-21 & 25-01-23 & 25-02-28 & 25-03-12'
        

but in 'VWC-Chadwick_Chapter-2_25-04-08'
Task = '23-02-16 & 23-02-23 & 23-03-08 & 23-03-09 & 23-03-16 & 23-03-17 & 23-03-24 & 22-12-07_M4 & 23-02-02_M1 & 23-01-23_M1 & 23-04-25 & 23-05-10 & 23-05-23_M1 & 23-07-07_M2 & 23-07-12_M1'

"""



filename =  'VWC-Chadwick_Chapter-2_25-04-23'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
fitsSubDir =  filename
data = pf.createDataTable(GlobalTable, fitsSubDir = fitsSubDir)


#%%%% Chadwick / Dimitriadis Model (3T3 Atcc / 3T3 C-OptoRhoA)

styleCellType =  {'optoRhoA':{'color': '#2986CC','marker':'o', 'label': '3T3 C-OptoRhoA\n(No light)'},
               'Atcc-2023':{'color': '#000000','marker':'o', 'label': '3T3 WT'},
               #'optoRhoA-NS':{'color': '#8fce00','marker':'o', 'label': '3T3 I-OptoRhoA'},
               }

# styleActType =  {'global':{'color': '#103551','marker':'o', 'label': 'Global Activation'},
#                'no light':{'color': '#2986CC','marker':'o', 'label': '3T3 C-OptoRhoA'},
#                }

styleDrugs = {'global':{'color': '#103551','marker':'o', 'label': 'Global Activation'},
               'no light':{'color': '#2986CC','marker':'o', 'label': '3T3 C-OptoRhoA'},
               }

celltypes = ['Atcc-2023'] 
# celltypes = ['optoRhoA' ] 

# celltypes = ['Atcc-2023'] #, 'Atcc-2023'] 

# celltypes = ['optoRhoA'] 
magFields = [5.0]
# magFields = [ 14.0, 15.0]
# 
activation = ['no light']
drugs = ['none']
# ramps = ['1_50', '2.5_50']

# activationfreq = [0, 3]
# firstAct = [-1, 1]

activationfreq = [0]
firstAct = [-1]

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 10),
            (data['compression duration'] == '1.5s'),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['normal field'].apply(lambda x : x in magFields)),
            # (data['ramp field'].apply(lambda x : x in ramps)),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['drug'].apply(lambda x : x in drugs)),
            
            ]
df = pf.filterDf(Filters, data)
# df = df.drop(df[(df['activation type'] == 'global') & (df['compNum'] == 1)].index)
# condCol, condCat = 'activation type', activation
df = pf.NLIcorr(df)

condCol, condCat = 'cell subtype', celltypes

pairs = [condCat]

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

palette_cond = pf.getSnsPalette(condCat, styleCellType)

styleDf = pd.DataFrame(styleCellType)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

#%%% Dimtriadis Model

# fullCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
#                     styleDict = styleCellType, condCol = condCol, mode = 'wholeCurve', 
#                     scale = 'lin', printText = False, returnData = 1, returnCount = 1)

# fig1, ax1, exportDf1, countDf1 = fullCurve[0]
# ax1.set_ylim(0, 15)
# plt.savefig(os.path.join(dirToSave, 'KvS_FullRange_WTvOpto.pdf'), dpi = 100)
# plt.show()

rangeCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleCellType, condCol = condCol, mode = '200_600', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig2, ax2, exportDf2, countDf2 = rangeCurve[0]
ax2.set_ylim(2, 9)
# plt.savefig(os.path.join(dirToSave, 'KvS_200_600_WTvOpt.pdf'), dpi = 100)
plt.show()

data_ff = taka.getFitsInTable(df, fitsSubDir, filter_fitID='_75')



#%%% Dimtriadis Model : thickness


viridis_palette = [
    "#440154",  # Dark Purple
    "#482878",  # Purple
    "#3E4A89",  # Deep Blue
    "#31688E",  # Blue
    "#26828E",  # Cyan-Blue
    "#1F9E89",  # Green-Cyan
    "#35B779",  # Bright Green
    "#6CBB3C",  # Yellow-Green
    "#B4DD2C",  # Yellow
    "#F4D22F",  # Bright Yellow
    "#F8D74D",  # Pale Yellow
    "#F8E89C",  # Very Pale Yellow
    "#FCEB8D",  # Lightest Yellow
    "#F7F4F9"   # Off-White
]

styleH0bin =  {1:{'color': viridis_palette[0],'marker':'o', 'label': '1-200nm'},
               2:{'color': viridis_palette[2],'marker':'o', 'label': '200-400nm'},
               3:{'color': viridis_palette[4],'marker':'o', 'label': '400-600nm'},
               4:{'color': viridis_palette[6],'marker':'o', 'label': '600-800nm'},
               5:{'color': viridis_palette[8],'marker':'o', 'label': '800-1000nm'},

               6:{'color': viridis_palette[10],'marker':'o', 'label': '1000-1200nm'},
               7:{'color': viridis_palette[12],'marker':'o', 'label': '1200-1400nm'},

               }


fullCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleH0bin, condCol = 'H0_Bin', mode = 'wholeCurve', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig1, ax1, exportDf1, countDf1 = fullCurve[0]
ax1.set_ylim(0, 12)
ax1.axvline(x=200, color='red', linestyle='--', linewidth=1)
ax1.axvline(x=600, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(os.path.join(dirToSave, 'KvS_FullRange_C-Opto_H0Bin.pdf'), dpi = 200)
plt.show()


# fullCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
#                     styleDict = styleH0bin, condCol = 'H0_Bin', mode = 'wholeCurve', 
#                     scale = 'lin', printText = False, returnData = 1, returnCount = 1)

# fig1, ax1, exportDf1, countDf1 = fullCurve[0]
# ax1.set_ylim(0, 12)
# ax1.axvline(x=200, color='red', linestyle='--', linewidth=1)
# ax1.axvline(x=600, color='red', linestyle='--', linewidth=1)
# plt.tight_layout()

# plt.savefig(os.path.join(dirToSave, 'KvS_FullRange_WT_H0Bin.pdf'), dpi = 200)
# plt.show()

#%%% Chadwick %f_15


y = 'Chadwick_%f_15_H0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
median = np.median(avgDf['Chadwick_%f_15_H0_log', 'mean'])

x_ticks = [0, 1]
ax.set_xticks(x_ticks, labels =labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
# plt.title('Cortical Thickness \n @ 15% Max Force (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + 'c-Opto_{:}.pdf').format(str(y)), dpi = 200)
plt.show()

#%%% E_f_<_400

y = 'E_f_<_400_log'

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)

x_ticks = [0, 1]
ax.set_xticks(x_ticks, labels =labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
# plt.title('Cortical Elasticity < 400pN (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + 'c-Opto_{:}.pdf').format(str('E_400pN_log')), dpi = 200)
plt.show()
#%%% VWC %f_100
plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}


y = 'bestH0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)


# plt.xlim(-0.5, 2.5)

plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
# plt.savefig((dirToSave + 'C-Opto_{:}.pdf').format(str(y)), dpi = 200)
plt.show()

#%%% E-eff
plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}


y = 'E_eff_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)

# plt.xlim(-0.5, 2.5)

plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + 'WT-c-Opto_{:}.pdf').format(str(y)), dpi = 200)
plt.show()

#%%% E vs H0

fig, axes = plt.subplots(2, 1, figsize = (21/SCALE_px_cm,21/SCALE_px_cm))

fig, ax, dataSlopes = pf.EvsH0_perCompression(fig, axes[0], df, condCat, condCol,
                                              hueType = condCol, metrics = 'Chadwick',
                                              colorScheme = 'white', palette = palette_cond)

fig, ax, avgDf = pf.EvH0_LogCellAvg(fig, axes[1],  avgDf, condCat, condCol, 
                                    hueType = 'condCol', metrics = 'Chadwick',
                                      colorScheme = 'white', palette = palette_cond)
# fig.suptitle(str(dates), **plotChars)

for ax in np.atleast_1d(axes):
    ax.tick_params(axis='both', labelsize=plotTicks['fontsize'], colors=plotTicks['color'])
plt.tight_layout()
plt.savefig((dirToSave + 'C-Opto_logE_400pNvChadwick_f15.pdf'))

plt.show()
plt.savefig((dirToSave + 'C-Opto_logE_400pNvChadwick_f15.pdf'))


# avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%% Fluctuations vs Thickness

fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm))

plottingParams = {'data':avgDf, 
                  'x' :('bestH0' , 'mean'),
                  'y':('ctFieldFluctuAmpli' , 'first'),
                  's':30
                    }

params, results = ufun.fitLineHuber(avgDf[('bestH0' , 'mean')], avgDf[('ctFieldFluctuAmpli' , 'first')])
pval = results.pvalues[1]
a, k = params[1], params[0]
x = np.linspace(100, 1200, 10)
y = a*x + k

eqnText = ''
eqnText += " Fit y = m * x + c\n".format(a, k)
eqnText += " y = {:.1e} * x + {:.1f}\n".format(a, k)
eqnText += " p-val = {:.1e}".format(pval)


ax = sns.scatterplot(label = eqnText, color = styleCellType[celltypes[0]]['color'],  **plottingParams)
plt.plot(x, y, color = styleCellType[celltypes[0]]['color'])

plt.legend(fontsize=12)
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
ax.set_ylim(0, 1000)
plt.show()
plt.savefig((dirToSave + 'Fluctuations_bestH0_{:}_{:}.pdf').format(celltypes, drugs), dpi = 200)

#%%% NLR

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'NLI_mod' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)



plt.xlim(-0.5, 2.5)

plt.ylim(-3, 3.5)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per Cell', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + 'Blebbi_COptoRhoA_{:}.pdf').format(y), dpi = 200)

#%%% NLR, Displot


disData = avgDf.reset_index()

disData.columns = [
    '_'.join(map(str, col)).strip() if isinstance(col, tuple) else str(col)
    for col in disData.columns
]

y = 'NLI_mod_mean'

plottingParams = {
    'data': disData,
    # 'col': f'{condCol}_first',  
    'y': y,      
    'stat':'percent',
    'bins':15,
    'hue':f'{condCol}_first',  
    'palette': palette_cond,
    'line_kws':{"linewidth": 3},
    'kde':True,
    }

fig, ax = plt.subplots(figsize=(10/SCALE_px_cm, 10/SCALE_px_cm))

ax = sns.histplot( **plottingParams)

ax.tick_params(axis='both', labelsize=plotTicks['fontsize'])

plt.axhline(y=0, color='r', linestyle='--')

plt.ylabel('Mean NLR per Cell (nm)', fontweight='bold', **plotChars)

plt.tight_layout()
# fig.title('Mean NLR per cell', fontweight='bold', **plotChars)
plt.ylim(-2.2, 1.2)
plt.xlim(0, 20)
plt.savefig((dirToSave + '{:}_WT-C-Opto_Displot.pdf').format( str(condCat)), dpi = 200)
plt.show()


#%%% H0 Histgram

plt.style.use('seaborn-v0_8')

y = 'Chadwick_%f_15_H0_log'

plottingParams = {
    'data': df,
    # 'col': f'{condCol}_first',  
    'y': y,      
    'stat':'percent',
    'bins':15,
    'hue': condCol,  
    'palette': palette_cond,
    'line_kws':{"linewidth": 3},
    'kde':True,
    }

fig, ax = plt.subplots(figsize=(10/SCALE_px_cm, 10/SCALE_px_cm))

ax = sns.histplot( **plottingParams)

ax.tick_params(axis='both', labelsize=plotTicks['fontsize'])

ax.set_yscale('log')

y_ticks = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
median_value = np.median(df[y])
plt.axhline(median_value, color='r', linestyle='-', label=f'Median: {median_value:.2f}')

plt.ylabel('Cortical Thickness (nm)',**plotChars)
plt.xlabel('% of Population',  **plotChars)

plt.title('Median Thickness = {:} nm'.format(np.round(10**median_value, 1)), fontweight='bold', **plotChars)
ax.get_legend().remove()

plt.savefig((dirToSave + '{:}_C-Opto_Histplot.pdf').format(y), dpi = 200)
plt.tight_layout()

# plt.savefig((dirToSave + 'WT-C-Opto_Displot.pdf').format( str(condCat)), dpi = 200)
plt.show()

#%%% E_histogram

plt.style.use('seaborn-v0_8')

y = 'E_f_<_400_log'

plottingParams = {
    'data': df,
    # 'col': f'{condCol}_first',  
    'y': y,      
    'stat':'percent',
    'bins':15,
    'hue': condCol,  
    'palette': palette_cond,
    'line_kws':{"linewidth": 3},
    'kde':True,
    }

fig, ax = plt.subplots(figsize=(10/SCALE_px_cm, 10/SCALE_px_cm))

ax = sns.histplot( **plottingParams)

ax.tick_params(axis='both', labelsize=plotTicks['fontsize'])

ax.set_yscale('log')

y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)

median_value = np.median(df[y])
plt.axhline(median_value, color='r', linestyle='-', label=f'Median: {median_value:.2f}')

plt.ylabel('Elasticity < 400pN (kPa)',  **plotChars)
plt.xlabel('% of Population',  **plotChars)

plt.title('Median Stiffness = {:} kPa'.format(np.round((10**median_value)/1000, 1)), fontweight='bold', **plotChars)
ax.get_legend().remove()

plt.savefig((dirToSave + '{:}_C-Opto_Histplot.pdf').format('E_f_400'), dpi = 200)
plt.tight_layout()

plt.show()

#%%% Rain plots, with distributions

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(15/SCALE_px_cm, 15/SCALE_px_cm))

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-3, 3)
plt.ylabel('', **plotChars)
plt.xlabel('', **plotChars)

plt.tight_layout()
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)

plt.savefig((dirToSave + 'WT-C-Opto_AvgNLRrainplot.png').format( str(condCat)), dpi = 100)
plt.show()



#%%% Plotnine jitter plots
#%%%% NLI - Scatterplot
measure = 'NLI_mod'
stat = 'mean'

pairs = [condCat]

plot = pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, pointSize = 2,
                     figsize=(20/SCALE_px_cm,12/SCALE_px_cm),
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1, 2))


plt.xticks([1,2], labels,color = 'black',fontsize = 11)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_WT-C-Opto.pdf'))

#%%%% H0

measure = 'bestH0_log'
stat = 'mean'
plot, df_median = pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, logScale = True,
                     pairs = pairs, stat = stat, palette = palette_cond, pointSize = 3,
                     figsize=(20/SCALE_px_cm, 13/SCALE_px_cm),
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels,color = 'black',fontsize = 11)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + '_WT-C-Opto_(Chad_f_15).pdf'))


#%%%% E_effective

measure = 'E_f_<_400_log'
stat = 'mean'
plot, df_medians= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure,  pointSize = 3,
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     figsize=(20/SCALE_px_cm, 10/SCALE_px_cm),
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels,color = 'black',fontsize = 11)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Elastic Modulus < 400pN (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()

plt.show()
plt.savefig((dirToSave + measure.split('<')[0] + '_WT-C-Opto '+measure.split('<')[1]+'.pdf'))



#%%% Plotting different stress-strain curves for different thicknesses

toPlot_f = df[df['H0_Bin'] == 2]

chosenDf = toPlot_f.groupby('cell subtype').apply(lambda g: g.sample(n=min(len(g), 5))).reset_index(drop=True)
chosenCells = chosenDf['cellID'].values

toPlot_f =  toPlot_f[toPlot_f['cellID'].apply(lambda x : x in chosenCells)]

fig, ax = pf.plotCellKS(toPlot_f, condCol = 'cell subtype', fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                   mode = 'wholeCurve')

# ax.set_ylim(0, 15)
# plt.savefig(os.path.join(dirToSave, 'KvS_FullRange_WTvOptovGlobal.pdf'), dpi = 100)
plt.show()


#%%  Calling Data - Global Activation with C-OptoRhoA 

"""
Task = '23-04-19 & 23-04-25 & 23-05-10'
        
"""

filename = 'VWC-Chadwick_Chapter-2_GlobalActivation_25-03-27'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
data = pf.createDataTable(GlobalTable, fitsSubDir = filename)


#%%%% Filters


styleActivation =  {'no light':{'color': '#2986CC','marker':'o', 'label': 'C-OptoRhoA\n(No Light)'},
               'global':{'color': '#144366','marker':'o', 'label': 'Global\nActivation'},
               }

celltypes = ['optoRhoA']
activation = ['no light', 'global']
magField = [14.0, 15.0]
drugs = ['none']
activationfreq = [0, 3]
firstAct = [-1, 1]
dates = [ '23-04-25', '23-05-10']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['compression duration'] == '1.5s'),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'activation type', activation
df = pf.NLIcorr(df)

styleDf = pd.DataFrame(styleActivation)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleActivation)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

#%%%% Chadwick / Dimitriadis Model (C-OptoRhoA (before and After Activation))

fullCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleActivation, condCol = condCol, mode = 'wholeCurve', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig1, ax1, exportDf1, countDf1 = fullCurve[0]
ax1.set_ylim(0, 15)
plt.savefig(os.path.join(dirToSave, 'KvS_FullRange_COptoGlobal.pdf'), dpi = 100)
plt.show()

rangeCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleActivation, condCol = condCol, mode = '200_600', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig2, ax2, exportDf2, countDf2 = rangeCurve[0]
ax2.set_ylim(2, 9)
plt.savefig(os.path.join(dirToSave, 'KvS_200_600_COptoGlobal.pdf'), dpi = 100)
plt.show()



#%%%% Plot NLImod

########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(9, 8))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + 'C-OptoRhoA_gLobalActivation_NLRrainplot.pdf').format( str(condCat)), dpi = 100)
plt.show()


######## cell average #########

fig, ax = plt.subplots(figsize = (9,8))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :8, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, colorScheme = 'white',
                             labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,2)
# fig.suptitle(str(dates), **plotChars)
plt.ylabel('Avg. NLR per Cell', **plotChars)
plt.xlabel(' ', **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + 'C-OptoRhoA_gLobalActivation_NLR_cellAvg.pdf').format(str(dates), str(condCat)))
plt.show()

######## coloured dates #########

# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : condCol, 
#                   'y' : 'NLI_mod',
#                   'order' : condCat,
#                   'linewidth' : 1,
#                   'size' :8, 
#                     }

# fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
#                                     hueType = 'date', labels = [], plottingParams = plottingParams,
#                                     plotChars = plotChars)

# # plt.ylim(-3, 3)
# fig.suptitle(str(dates), **plotChars)
# plt.yticks(**plotTicks)
# plt.xticks(**plotTicks)
# plt.tight_layout()
# plt.savefig((dirToSave + '(1f)_{:}_{:}_NLImodPLot-Dates.png').format(str(dates), str(condCat)))


#%%% Chadwick %f_10

y = 'Chadwick_%f_15_H0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':6,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
# plt.yscale('log')
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

# y_labels = np.asarray([100 ,250, 500, 1000, 1500])
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

y_labels = np.asarray([0,250, 500, 750, 1000, 1250, 1500])
# y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_labels, labels =y_labels,**plotTicks)

x_labels = ['5mT\n~50pN', '15mT\n~500pN']
x_ticks = [0,1]
ax.set_xticks(x_ticks, labels =x_labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness \n @ 15% Max Force (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()

#%%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# # plt.yscale('log')
# # plt.ylim(100, 30000)
# # y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                             labels = labels, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 2000, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))
plt.show()


####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                             labels = labels, plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 2000, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()



#%%% E vs H0

fig, axes = plt.subplots(2, 1, figsize = (21/SCALE_px_cm,21/SCALE_px_cm))

fig, ax, dataSlopes = pf.EvsH0_perCompression(fig, axes[0], df, condCat, condCol, hueType = condCol,
                                              colorScheme = 'white', palette = palette_cond)

fig, ax, avgDf = pf.EvH0_LogCellAvg(fig, axes[1],  avgDf, condCat, condCol, hueType = 'condCol',
                                      colorScheme = 'white', palette = palette_cond)
# fig.suptitle(str(dates), **plotChars)

for ax in np.atleast_1d(axes):
    ax.tick_params(axis='both', labelsize=plotTicks['fontsize'], colors=plotTicks['color'])
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + 'C-Opto-GlobalAct_logAvgE_400pNvChadwick_f15.pdf').format(str(dates), str(condCat)))


# avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%% Plotnine paired plots
#%%%% H0
dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())

measure = 'Chadwick_%f_15_H0_log'
stat = 'mean'
plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure, logScale = True,
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness \n @ 15% Max Force (nm)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + 'C-Opto_{:}_{:}-{:}_PairedPlot.pdf').format(str(condCat), measure, stat))

# Normalized
plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2.5))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Cortical Thickness', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + 'C-Opto_{:}_{:}-{:}_NormPairedPlot.pdf').format(str(condCat), measure, stat))
#%%%% E_f_<_400


measure = 'E_f_<_400_log'
stat = 'mean'
plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure, logScale = True,
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks)



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Elasticity < 400pN (kPa)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + 'C-Opto_{:}_{:}-{:}_PairedPlot.pdf').format(str(condCat), 'E_400_', stat))

# Normalized
plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2.5))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Elasticity', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + 'C-Opto_{:}_{:}-{:}_NormPairedPlot.pdf').format(str(condCat), 'E_400_', stat))

#%%%% NLI - Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test= 'less',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1,2))



plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)\n', fontweight='bold', **plotChars)
plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', lw = 2, alpha  =0.4)

plt.show()
plt.savefig((dirToSave + measure + '_C-OptoRhoA_'+str(activation)+'_Paired.pdf'))

#%%%% NLR - Normalized Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                          figsize = (10/SCALE_px_cm,10/SCALE_px_cm),
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.5, 2.5))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')

plt.xlabel('')
plt.title('Normalized NLR per Cell', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_C-OptoRhoA_'+str(activation)+'_NormPaired.pdf'))
#%%% H0 @ 10% Force vs. Compression

avgDf_Plot = avgDf[(avgDf[('compNum', 'count')] > 4)]

dfPairs, pairedCells = pf.dfCellPairs(avgDf_Plot)
cellsChosen = np.unique(dfPairs[('dateCell', 'first')].values)
# toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]

random_10 = (pd.Series(cellsChosen).sample(8)).values

toPlot = df[df['dateCell'].apply(lambda x : x in random_10)]

toPlot.loc[toPlot[condCol] == 'global', 'compNum'] += 6
# toPlot = toPlot.drop(toPlot[(toPlot['cellID'] == '22-10-06_M5_P3_C3') & (toPlot['compNum'] == 1)].index)

split_point = 6
y = 'Chadwick_%f_15_H0'
colorblind_type = "Deuteranomaly"
palette = distinctipy.get_colors(len(toPlot['dateCell'].unique()), colorblind_type=colorblind_type)

toPlot['new_compNum'] = toPlot['compNum']
toPlot['new_compNum'].loc[toPlot['compNum'] < split_point] = 0
toPlot['new_compNum'].loc[toPlot['compNum'] >= split_point] = toPlot['compNum'] - split_point

# Apply this to each cellID group
toPlot = toPlot.groupby('dateCell', group_keys=False).apply(pf.normalize_by_first_five, y)

y_new = y + '_norm'

plottingParams1 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'dateCell',
                  'kind':'point',
                  'markersize' : 15,
                  'palette':palette,
                  'marker':'.',
                  'linewidth': 2,
                  'errorbar':'sd',
                  'errwidth':2,
                  'alpha':0.4,
                  'aspect':1, 
                  'capsize':.05
                 }


plottingParams2 = {'data':toPlot,
                  'x':'new_compNum', 
                   
                  'hue':'dateCell',
                  'marker': None,
                  'linewidth':2,
                  'palette':palette,
                  'alpha':0.4,
                 }

plottingParams3 = {'data':toPlot,
                  'x':'new_compNum', 
                  'marker': 'o',
                  'linewidth':2,
                  'color':'k',
                 }
                 

g = sns.catplot(y = y_new, legend_out = False,  **plottingParams1)
g.fig.set_size_inches(8.6, 2.8)
ax1 = g.ax
ax1 = sns.lineplot(y = y_new, legend = False,  **plottingParams2)
ax1 = sns.lineplot(y = y_new, legend = False,  **plottingParams3)


# ax2 = ax1.twinx()
# ax2 = sns.lineplot(y = y_new, legend = False, **plottingParams3)

for i in range(10):
    plt.axvline(x= 0.1 + i, color='blue', linestyle='-', 
                    ymin=0, ymax=0.1, linewidth=2)
    
plt.legend(labelcolor='linecolor', bbox_to_anchor=(1.05, 1))


ax1.set_ylim(0, 2.5)
# ax2.set_ylim(0, 3)

plt.xlim(-0.1, 10)

 
for ax in [ax1, ax2]:

    ax.tick_params(axis='both', labelsize = plotTicks['fontsize'])
   
    ax.set_xlabel('Compressions Post-Activation', **plotChars)

# ax1.set_ylabel('Cortical Thickness (nm)', **plotChars)

ax1.set_ylabel('Normalized\nCortical Thickness', **plotChars)


plt.tight_layout()
plt.savefig((dirToSave + y_new + 'vsComp_Sequence-{:}_C-Opto.pdf').format(str(condCat) ), dpi = 100)

plt.show()


#%%% Cell-dependent NLR Variability

plotter = df

plotter = plotter.groupby('cellID', \
                          group_keys=False).apply(pf.NLR_normalize_by_first_six,
                                                  'NLI_mod', clip = (0, 6),
                                                  col = 'compNum')


plotter2 = avgDf[('cellID', 'first')]
colorblind_type = "Deuteranomaly"

unique_cellIDs = plotter['cellID'].unique()
Ncells = len(unique_cellIDs)
palette = distinctipy.get_colors(Ncells, colorblind_type=colorblind_type)

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))
cellID_to_color = {cell: palette[i] for i, cell in enumerate(unique_cellIDs)}

plottingParams = {'data':plotter, 
                  'x' : ('cellID'), 
                  'y' : ('NLI_mod_norm'),
                  'hue':'cellID',
                  'palette':cellID_to_color,
                  's':4
                 }

sns.swarmplot(**plottingParams, ax = ax)
for collection in ax.collections:
    collection.set_edgecolor('k')
    collection.set_linewidth(0.5) 


avg_var = plotter['NLI_mod_std'].mean()

for _, row in plotter.iterrows():
    plt.errorbar(x=row['cellID'], y=row['NLI_mod_ref'], yerr=row['NLI_mod_std'],
                 fmt="none", color=cellID_to_color[row['cellID']],
                 capsize=5, alpha=0.4, lw = 0.3,  capthick=1)
    
    
text = 'Average STD of Cells = {:.2f}'.format(avg_var)

plt.text(0, 3, text, fontsize=12)
x_labels = ['C'+str(i+1) for i in range(Ncells)]
ax.set_xticks(ax.get_xticks(), labels =x_labels, rotation = 90, **plotTicks)
plt.ylim(-3, 3.5)
ax.get_legend().remove()
plt.savefig((dirToSave + '{:}_NLI_StandardDev.pdf').format(condCat), dpi = 200)

#%%% Cell-dependent NLR evolution

plotter = avgDf

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))

plottingParams = {'data':plotter, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'hue':('cellID','first'),
                  'palette':cellID_to_color,
                  's':10,
                 
                 }

sns.swarmplot(**plottingParams, ax = ax)
plt.ylim(-3, 3.5)

var_pop = plotter[('NLI_mod', 'mean')].var()
text = 'STD of the Population = {:.2f}'.format(var_pop)
plt.text(0, 3, text, fontsize=12)
ax.get_legend().remove()
plt.savefig((dirToSave + '{:}_NLI_StdDevofMean.pdf').format(condCat), dpi = 200)



#%%% E_400pN vs. Compression

y = 'E_f_<_400'

y_new = y + '_norm'

toPlot = toPlot.groupby('dateCell', group_keys=False).apply(pf.normalize_by_first_five, y)

plottingParams1 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'dateCell',
                  'kind':'point',
                  'markersize' : 15,
                  'palette':palette,
                  'marker':'.',
                  'linewidth': 2,
                  'errorbar':'sd',
                  'errwidth':2,
                  'alpha':0.4,
                  'aspect':1, 
                  'capsize':.05
                 }


plottingParams2 = {'data':toPlot,
                  'x':'new_compNum', 
                   
                  'hue':'dateCell',
                  'marker': None,
                  'linewidth':2,
                  'palette':palette,
                  'alpha':0.4,
                 }

plottingParams3 = {'data':toPlot,
                  'x':'new_compNum', 
                  'marker': 'o',
                  'linewidth':2,
                  'color':'k',
                 }
                 

g = sns.catplot(y = y_new, legend_out = False,  **plottingParams1)
g.fig.set_size_inches(8.6, 2.8)
ax1 = g.ax
ax1 = sns.lineplot(y = y_new, legend = False,  **plottingParams2)
ax1 = sns.lineplot(y = y_new, legend = False,  **plottingParams3)


# ax2 = ax1.twinx()
# ax2 = sns.lineplot(y = y_new, legend = False, **plottingParams3)

for i in range(10):
    plt.axvline(x= 0.1 + i, color='blue', linestyle='-', 
                    ymin=0, ymax=0.1, linewidth=2)
    
plt.legend(labelcolor='linecolor', bbox_to_anchor=(1.05, 1))


# ax2.set_ylim(0, 3)

plt.xlim(-0.1, 10)

 

ax1.set_yscale('log')
ax1.set_ylim(0, 10)

ax1.tick_params(axis='both', labelsize = plotTicks['fontsize'])
   
ax1.set_xlabel('Compressions Post-Activation', **plotChars)

# ax1.set_ylabel('Cortical Thickness (nm)', **plotChars)

ax1.set_ylabel('Normalized\nElasticity', **plotChars)


plt.tight_layout()
plt.savefig((dirToSave  + 'E_400pN_vsComp_Sequence-{:}_C-Opto.pdf').format(str(condCat) ), dpi = 100)

plt.show()
plt.show()

#%%% all plot types

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())


measure = 'NLI_mod'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'less', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1, 2))


plt.xticks([1, 2], labels, fontsize=15, color = 'black')
plt.yticks(**plotTicks)
plt.ylabel('Avg. NLR per Cell', **plotChars)
plt.xlabel(' ', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + 'C-OptoRhoA_globalActivation_PairedPlot_60sFreq.pdf'))



measure = 'E_eff_log'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (2, 4.5))


plt.yscale('log')
plt.xticks([1, 2], labels, **plotTicks)
y_labels = [100, 500, 2000, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
plt.yticks(y_ticks, labels =y_labels,**plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'H0_vwc_Full'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1500))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


measure = 'ctFieldThickness'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'ctFieldFluctuAmpli'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'NLI_mod'
stat = 'std'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))

#%%%%  Pointplots / Pairedplots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)


testH0 = 'two-sided'
testE = 'less'
testNli = 'two-sided'
stats = 'mean'

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# ax.get_legend().remove()
plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))


plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1200
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()



plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('ctFieldThickness', 'first'),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,1200), 
                                          pairs = pairs, normalize = False, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_ctFieldThickness.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}__ctFieldThickness-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff_log', 'mean'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (2,4.5), 
                                          pairs = pairs, normalize = False, marker = 'mean',
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.yscale('log')
y_labels = [100, 500, 2000, 5000, 10000]
y_ticks = np.log10(np.asarray(y_labels))
plt.yticks(y_ticks, labels =y_labels,**plotTicks)
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
#                                           pairs = pairs, normalize = True, marker = 'mean',
#                                           test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

# plottingParams = {'x' : (condCol, 'first'), 
#                   'y' : ('E_norm', 'mean'),
#                   'linewidth' : 1,
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                    }


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,5000), 
#                                           pairs = pairs, normalize = False, marker = 'wAvg',
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)

# plt.show()
# y_ticks = [100, 250, 500, 1000, 2500, 5000]
# ax.set_yticks(y_ticks, labels =y_ticks, fontsize=25, color = fontColour)
# plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_E-normPointplot.png').format(str(dates), str(condCat), stats))


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
#                                           pairs = pairs, normalize = True, marker = 'wAvg',
#                                           test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_E-normPointplot-Normalised.png').format(str(dates), str(condCat), stats))





# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
#                                           pairs = pairs, normalize = True, marker = stats,
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('surroundingDz', 'median'),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-1000,1200), 
                                          pairs = pairs, normalize = False, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_surroundingDz.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-2,3), 
                                          pairs = pairs, normalize = True, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}__surroundingDz-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()

#%%%% Pointplots with errorbars
dfPairs, pairedCells = pf.dfCellPairs(avgDf)
dfPairs = df[df['dateCell'].apply(lambda x : x in pairedCells)]

measure = 'NLI_mod'
plottingParams = {'x' : condCol, 
                  'y' : measure,
                  'data' : dfPairs,
                  'hue' : 'dateCell',
                  'dodge' : True,
                  'palette' : palette_cell,
                  'errorbar' : 'se'
                   }

fig, ax = plt.subplots(figsize = (13,10))

fig.patch.set_facecolor('black')

ax = sns.pointplot(**plottingParams)

plt.xticks(**plotChars)
plt.yticks(**plotChars)
plt.legend(fontsize = 15, labelcolor='linecolor', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(10a)_{:}_{:}_{:}_Errorbar.png').format(str(dates), str(condCat),  str(measure)))

#%%%% E vs H0

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

dfPairs = df[df['dateCell'].apply(lambda x : x in pairedCells)]

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, dfPairs, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 16, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH-dfPairs_Conditions.png').format(str(dates), str(condCat)))

# dfPairs, pairedCells = pf.dfCellPairs(avgDf)
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax, df = pf.EvH0_LogCellAvg(fig, ax, dfPairs, condCat, condCol, palette = palette_cond)
# plt.legend(fontsize = 16, ncol = len(condCat))
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH-dfPairs_wAvg.png').format(str(dates), str(condCat)))

avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%%% Thickness vs. time

measure = 'surroundingThickness'
activationTime = []

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')
flatui = ["#000000", "#0000ff"]

x = (df['compNum']-1)*20
ax = sns.lineplot(x = x, y = measure, data = df, hue = condCol)

plt.ylim(0,1200)
plt.xlim(0,140)

# plt.savefig(dirToSave + '/Thickness/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'.png')

plt.xticks(**plotChars)
plt.yticks(**plotChars)
plt.show()



#%% Calling Data - At Beads Polarization with 60s Frequency
"""
Task =  '23-05-23 & 23-07-12'

fitsSubDir = 'VWC-Chadwick_Chapter-2_AtBeadsPolarisation_60sFreq_25-04-24'

"""


filename = 'VWC-Chadwick_Chapter-2_AtBeadsPolarisation_60sFreq_25-04-24'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
fitsSubDir =  filename
data = pf.createDataTable(GlobalTable, fitsSubDir = fitsSubDir)


#%%% Filter Data

styleActivation =  {'no light':{'color': '#2986CC','marker':'o', 'label': 'C-OptoRhoA\n(No Light)'},
               'at beads':{'color': '#144366','marker':'o', 'label': 'Polarized Rear'},
               }

celltypes = ['optoRhoA']
activation = ['no light', 'at beads']
magField = [15.0]
drugs = ['none', 'activation']
activationfreq = [0, 3]
firstAct = [-1, 1]
dates = ['23-05-23', '23-07-12']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 10),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['compression duration'] == '1.5s'),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'activation type', activation

styleDf = pd.DataFrame(styleActivation)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleActivation)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 25}
plotTicks = {'color' : '#000000', 'fontsize' : 22}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]


#%%% NLR

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'NLI_mod' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)



plt.xlim(-0.5, 2.5)

plt.ylim(-3, 3.5)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per Cell', fontweight='bold', **plotChars)
plt.show()
# plt.savefig((dirToSave + 'AtBeads_60s_COptoRhoA_{:}.pdf').format(y), dpi = 200)

#%%% VWC %f_100
plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}


y = 'bestH0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)


plt.xlim(-0.5, 2.5)

plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
# plt.savefig((dirToSave + 'WT-c-Opto_{:}.pdf').format(str(y)), dpi = 200)
plt.show()

#%%% E-eff
plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}


y = 'E_eff_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,
                                    palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)

plt.xlim(-0.5, 2.5)

plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()
# plt.savefig((dirToSave + 'WT-c-Opto_{:}.pdf').format(str(y)), dpi = 200)
plt.show()

#%%% Plotnine paired plots
#%%%% NLI - Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1,2))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)\n', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%%% BestH0 - Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)\n ', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%%% E_eff - Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, dataSlopes = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol, hueType = condCol,
                                              colorScheme = 'white', palette = palette_cond)
fig.suptitle(str(dates), **plotChars)
plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))

fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'condCol',
                                      colorScheme = 'white', palette = palette_cond)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.png').format(str(dates), str(condCat)))



# avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%% Calling Data - At Beads Polarization with 20s Frequency
"""

Task =  '23-01-23 & 23-02-02_M1 & 23-02-02_M2 & 22-12-07_M4 & 22-12-07_M5'

        
fitsSubDir = 'VWC-Chadwick_Chapter-2_AtBeadsPolarisation_20sFreq_25-04-24'

"""


filename = 'VWC-Chadwick_Chapter-2_AtBeadsPolarisation_20sFreq_25-04-24'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
fitsSubDir =  filename
data = pf.createDataTable(GlobalTable, fitsSubDir = fitsSubDir)


#%%% Filter Data

styleActivation =  {'no light':{'color': '#2986CC','marker':'o', 'label': 'C-OptoRhoA\n(No Light)'},
               'at beads':{'color': '#144366','marker':'o', 'label': 'Polarized Rear'},
               }

celltypes = ['optoRhoA']
activation = ['no light', 'at beads']
magField = [14.0, 15.0]
drugs = ['none', 'activation']
activationfreq = [0, 1]
firstAct = [-1, 1]
dates = ['23-01-23', '23-02-02', '22-12-07']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 10),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['compression duration'] == '1.5s'),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'activation type', activation

styleDf = pd.DataFrame(styleActivation)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleActivation)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 25}
plotTicks = {'color' : '#000000', 'fontsize' : 22}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]


#%%% NLR

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'NLI_mod' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)



plt.xlim(-0.5, 2.5)

plt.ylim(-3, 3.5)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per Cell', fontweight='bold', **plotChars)
plt.show()
# plt.savefig((dirToSave + 'AtBeads_60s_COptoRhoA_{:}.pdf').format(y), dpi = 200)

#%%% Plotnine paired plots
#%%%% NLI - Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1,2))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)\n', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%%% BestH0 - Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)\n ', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%%% E_eff - Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, dataSlopes = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol, hueType = condCol,
                                              colorScheme = 'white', palette = palette_cond)
fig.suptitle(str(dates), **plotChars)
plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))

fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'condCol',
                                      colorScheme = 'white', palette = palette_cond)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.png').format(str(dates), str(condCat)))



# avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)


#%%  Calling Data - Global Activation with I-OptoRhoA

"""
Task = '24-12-14 & 24-12-20 & 25-01-14 & 25-01-21 & 25-01-23 & 25-02-28 & 25-03-12'
        
"""

filename = 'VWC-Chadwick_Chapter-2_I-OptoRhoA_25-03-28'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/FiguresForManuscript/Chapter_2/'
data = pf.createDataTable(GlobalTable)
fitsSubDir = filename


#%%%% Filters


styleActivation =  {#'none':{'color': '#808080','marker':'o', 'label': 'I-OptoRhoA\n(Non-induced)'},
                'doxy':{'color': '#8c198c','marker':'o', 'label': 'No Light'},
               'doxy_act':{'color': '#330033','marker':'o', 'label': 'Global\nRhoA Activation'},
               }

celltypes = ['optoRhoA-NS']
activation = ['no light', 'global']
magField = [5.0]
drugs = ['doxy', 'doxy_act']
activationfreq = [0, 3]
firstAct = [-1, 1]
dates = ['24-12-14', '24-12-20', '25-01-14', '25-01-21', '25-01-23']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 10),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs

styleDf = pd.DataFrame(styleActivation)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleActivation)

pairs = [['doxy', 'doxy_act']]

plotChars = {'color' : '#000000', 'fontsize' : 15}
plotTicks = {'color' : '#000000', 'fontsize' : 14}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

#%%% NLI vs. Compression

# avgDf = pf.createAvgDf(df, condCol)
# avgDf = avgDf[(avgDf[('compNum', 'count')] > 5)]

# dfPairs, pairedCells = pf.dfCellPairs(avgDf)

# cellsChosen = dfPairs['cellID'].values
# toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]

# toPlot['compNum'][toPlot[condCol] == 'doxy_act'] = toPlot['compNum'] + 5

# plottingParams1 = {'data':toPlot,
#                   'x':'compNum', 
#                   'y':'H0_vwc_Full', 
#                   'hue':'dateCell',
#                   'marker':'o',
#                   'alpha':0.5,
#                  }

# plottingParams2 = {'data':toPlot,
#                   'x':'compNum', 
#                   'y':'H0_vwc_Full', 
#                   'marker':'o',
#                   'color':'black'
#                  }


# fig, ax = plt.subplots(figsize = (14/SCALE_px_cm,12/SCALE_px_cm))
# ax = sns.lineplot( **plottingParams1)
# ax = sns.lineplot(**plottingParams2)

# N = 2  # You can change N to any value
# for i in range(0, 2):
#     plt.axvline(x= 6 + (i*3), color='blue', linestyle='-', 
#                 ymin=0, ymax=0.1, linewidth=2)
    
    
# ax.get_legend().remove()
# plt.xticks(**plotTicks)
# plt.yticks(**plotTicks)
# plt.ylim(0, 1200)
# plt.ylabel('')
# plt.xlabel('Compression No.')
# plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)
# plt.savefig(os.path.join(dirToSave, 'H0vComp_R298_Comp5_IOptoGlobal.pdf'), dpi = 100)

# plt.show()

#%%% Chadwick / Dimitriadis Model (I-OptoRhoA (before and After Activation))

fullCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleActivation, condCol = condCol, mode = 'wholeCurve', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig1, ax1, exportDf1, countDf1 = fullCurve[0]
ax1.set_ylim(0, 15)
# plt.savefig(os.path.join(dirToSave, 'KvS_FullRange_IOptoGlobal.pdf'), dpi = 100)
plt.show()

rangeCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleActivation, condCol = condCol, mode = '150_500', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig2, ax2, exportDf2, countDf2 = rangeCurve[0]
ax2.set_ylim(2,10)
# plt.savefig(os.path.join(dirToSave, 'KvS_200_600_IOptoGlobal.pdf'), dpi = 100)
plt.show()

#%%% Plotnine jitter plots
#%%%% NLI
measure = 'NLI_mod'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, pointSize = 2.5,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1, 2))

plt.xticks([1,2,3], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + measure + 'I-OptoRhoA_GlobalActivation.pdf'))

#%%%% H0

measure = 'bestH0_log'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, logScale = True,
                     pairs = pairs, stat = stat, palette = palette_cond, pointSize = 2.5,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)

plt.xticks([1,2,3], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + 'I-OptoRhoA_GlobalActivation.pdf'))


#%%%% E_effective

measure = 'E_eff_log'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure,  pointSize = 2.5,
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)

plt.xticks([1,2,3], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + 'I-OptoRhoA_GlobalActivation.pdf'))

#%%% Plotnine paired plots
#%%%% NLI - Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1,2))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)\n', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%%% BestH0 - Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)\n ', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%%% E_eff - Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_GlobalActivation_Paired.pdf'))

#%%% Thickness vs. time

measure = 'surroundingThickness'
activationTime = []

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')
flatui = ["#000000", "#0000ff"]

x = (df['compNum']-1)*20
ax = sns.lineplot(x = x, y = measure, data = df, hue = condCol)

plt.ylim(0,1200)
plt.xlim(0,140)

# plt.savefig(dirToSave + '/Thickness/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'.png')

plt.xticks(**plotChars)
plt.yticks(**plotChars)
plt.show()

#%%% NLR / delta NLR vs. Angle from activation

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

toPlot = dfPairs.dropna(subset=[('angle_beads', 'first')])
# toPlot = toPlot.drop(toPlot[(toPlot['activation type'] != 'side')].index)

N_cols = distinctipy.get_colors(len(toPlot[('dateCell', 'first')].unique()))

plottingParams = {'data' : toPlot,
                  'x' : ('angle_beads', 'first'),
                  'y' : ('NLI_mod', 'diff'),
                  'hue' : ('dateCell', 'first'),
                  'palette' : N_cols,
                  's' : 100
                   }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.NLRvAngle(fig, ax, toPlot, condCat, condCol, pairedCells, colorScheme = 'white',
                pairs = pairs,  plotType = 'delta', plottingParams = plottingParams,  
                palette = palette_cond, plotChars = plotChars)


# sns.scatterplot(**plottingParams)

#%% Calling data - All Data, Y27, Blebbistatin, C-OptoRhoA + Global Activation

"""
Task = '22-12-07 & 23-03-24 & 23-02-02 & 23-04-25 & 23-05-10'
        
"""

filename = 'VWC-Chadwick_Chapter-2_Drugs-GlobalActivation_25-03-28'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
fitsSubDir = 'VWC-Chadwick_Chapter-2_Drugs-GlobalActivation_25-03-28'

data = pf.createDataTable(GlobalTable, fitsSubDir = fitsSubDir)


#%%% Filters For Y27 Dose Response

option = 'blebbi'

if option == 'blebbi':
    styleDict =  {'dmso_10':{'color': '#7f7f7f','marker':'o', 'label': 'DMSO'},
                  'blebbi_10':{'color': '#38761D','marker':'o', 'label': '10µM\nBlebbi'},
                   }
    dates = ['23-03-24']
    drugs = ['dmso_10', 'blebbi_10']
    pairs = [['blebbi_10', 'dmso_10']]
    
elif option == 'Y27':
    styleDict =  {'none':{'color': '#2986CC','marker':'o', 'label': 'C-OptoRhoA\n(No Light)'},
                     'Y27_10':{'color': '#a86697','marker':'o', 'label': '10µM\nY27'},
                     'Y27_50':{'color': '#753464','marker':'o', 'label': '50µM\nY27'},
                     # 'Y27_1':{'color': '#f8ae85','marker':'o', 'label': '1µM Y27'},
                   }
    drugs = ['none',  'Y27_10', 'Y27_50'] #, 'none', 'activation']
    dates = ['23-02-02', '23-04-25', '22-12-07']
    pairs = [['Y27_50', 'Y27_10'], ['none', 'Y27_10'], ['Y27_50', 'none']]



celltypes = ['optoRhoA']

magField = [14.0, 15.0]
activationfreq = [0]
firstAct = [-1]
# drugs = ['none', 'Y27_10', 'Y27_50']
# dates = ['23-02-02']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            # (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 10),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['compression duration'] == '1.5s'),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
df = pf.NLIcorr(df)

df['activity'] = df['ctFieldFluctuAmpli'] / df['ctFieldThickness']

styleDf = pd.DataFrame(styleDict)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleDict)

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

#%%%% Chadwick / Dimitriadis Model (C-OptoRhoA (before and After Activation))

# fullCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
#                     styleDict = styleDictBlebbi, condCol = condCol, mode = 'wholeCurve', 
#                     scale = 'lin', printText = False, returnData = 1, returnCount = 1)

# fig1, ax1, exportDf1, countDf1 = fullCurve[0]
# ax1.set_ylim(0, 15)
# plt.savefig(os.path.join(dirToSave, 'KvS_FullRange_Y27_COpto.pdf'), dpi = 200)
# plt.show()

rangeCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleDict, condCol = condCol, mode = '200_600', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig2, ax2, exportDf2, countDf2 = rangeCurve[0]
ax2.set_ylim(2, 9)
# plt.savefig(os.path.join(dirToSave, 'KvS_200_600_{:}_COpto.pdf'.format(option)), dpi = 200)
plt.show()

#%%% Fluctuations vs Thickness

fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm))

plottingParams = {'data':avgDf, 
                  'x' :('bestH0' , 'mean'),
                  'y':('ctFieldFluctuAmpli' , 'first'),
                  's':30
                    }

params, results = ufun.fitLineHuber(avgDf[('bestH0' , 'mean')], avgDf[('ctFieldFluctuAmpli' , 'first')])
pval = results.pvalues[1]
a, k = params[1], params[0]
x = np.linspace(100, 1200, 10)
y = a*x + k

eqnText = ''
eqnText += " Fit y = m * x + c\n".format(a, k)
eqnText += " y = {:.1e} * x + {:.1f}\n".format(a, k)
eqnText += " p-val = {:.3f}".format(pval)


ax = sns.scatterplot(label = eqnText, color = styleDict[drugs[0]]['color'],  **plottingParams)
plt.plot(x, y, color = styleDict[drugs[0]]['color'])

plt.legend(fontsize=12)
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
ax.set_ylim(0, 1000)
plt.show()
plt.savefig((dirToSave + 'Fluctuations_bestH0_{:}_{:}.pdf').format(celltypes, drugs), dpi = 200)

#%%% Chadwick %f_15

y = 'Chadwick_%f_15_H0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':5,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True,
                    }



fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
plt.yscale('log')

fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,test = 'non-param',
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)


x_ticks = [0,1]
ax.set_xticks(x_ticks, labels = labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness \n @ 15% Max Force (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + '{:}-C-Opto_{:}.pdf').format(option, str(y)), dpi = 200)


#%%% E_effective

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'E_eff_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =(y_labels)/1000,**plotTicks)

# x_labels = ['5mT\n~50pN', '15mT\n~500pN']
# x_ticks = [0,1]
# ax.set_xticks(x_ticks, labels =x_labels,**plotTicks)

plt.xlim(-0.5, 2.5)
plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + '{:}_COptoRhoA_{:}.pdf').format(option, y), dpi = 200)


#%%% Activity (Fluctuations / Thickness)

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

dfPlot = df.drop_duplicates(subset = ['cellID'])

y = 'activity' 

plottingParams = {'data':dfPlot, 
                  'x' : (condCol), 
                  'y' : (y),
                  'order' : condCat,
                  's':3,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm))

fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)


# plt.ylim(0, 1.5)

plt.xlim(-0.5, 2.5)

plt.ylabel('')
plt.xlabel('')
plt.title('Activity (Fluctuations / Thickness)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '{:}_COptoRhoA_{:}.pdf').format(option, y), dpi = 200)


#%%% VWC %F_100

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'bestH0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =(y_labels),**plotTicks)

plt.xlim(-0.5, 2.5)

plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness (nm)', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + '{:}_COptoRhoA_{:}.pdf').format(option, y), dpi = 200)

#%%% NLR

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'NLI_mod' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)



plt.xlim(-0.5, 2.5)
plt.axhline(y= 0, color = 'red', lw = 1, alpha  =0.4)

plt.ylim(-3, 3.5)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per Cell', fontweight='bold', **plotChars)
plt.show()
# plt.savefig((dirToSave + '{:}_COptoRhoA_{:}.pdf').format(option, y), dpi = 200)

#%%% E vs H0

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, dataSlopes = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol, hueType = condCol,
                                                      colorScheme = 'white', palette = palette_cond,
                                                      metrics = 'Chadwick')
fig.suptitle(str(dates), **plotChars)
plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize=(18/SCALE_px_cm, 10/SCALE_px_cm))
fig, ax, avgDf = pf.EvH0_LogCellAvg(fig, ax,  avgDf, condCat, condCol, 
                                    hueType = 'condCol', metrics = 'Chadwick',
                                      colorScheme = 'white', palette = palette_cond)

# fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.pdf').format(str(dates), str(condCat)))

avgDf =  pf.createAvgDf(dataSlopes, condCol, dataFluoPath = None, e_norm = True)

#%%% E_normalized

y = 'E_norm_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True,
                    }



fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)

fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,test = 'non-param',
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

x_ticks = [0,1,2]
ax.set_xticks(x_ticks, labels = labels,**plotTicks)

y_labels = np.asarray([100, 500, 1000, 3000, 10000, 25000])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =(y_labels)/1000,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '{:}-C-Opto_{:}.pdf').format(option, str(y)), dpi = 200)

#%%% NLI_Corr

y = 'NLI_corr' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'first'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True,
                    }



fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)

fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs,test = 'non-param',
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

x_ticks = [0,1,2]
ax.set_xticks(x_ticks, labels = labels,**plotTicks)


plt.ylabel('')
plt.xlabel('')
plt.title('NLI-corr', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '{:}-C-Opto_{:}.pdf').format(option, str(y)), dpi = 200)

#%%% Plotnine paired plots
#%%%% NLI
measure = 'NLI_mod'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, 
                         pairs = pairs, stat = stat, palette = palette_cond_y27,pointSize = 2.5,
                         plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1,2))


plt.xticks([1,2,3, 4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + 'C-OptoRhoA_Drugs-DoseResponse-Y27.pdf'))

#%%%% H0

measure = 'bestH0_log'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, logScale = True,
                     pairs = pairs, stat = stat, palette = palette_cond_y27, pointSize = 2.5,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2,3, 4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + 'C-OptoRhoA_Drugs-DoseResponse-Y27.pdf'))


#%%%% E_effective

measure = 'E_eff_log'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure,  pointSize = 2.5,
                     pairs = pairs, stat = stat, palette = palette_cond_y27, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2,3, 4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + 'C-OptoRhoA_Drugs-DoseResponse-Y27.pdf'))

#%%% Filters for blebbistatin Experiments

styleDict =  {'blebbi_10':{'color': '#35F478','marker':'o', 'label': '10µM\nBlebbistatin'},
              'dmso_10':{'color': '#2986CC','marker':'o', 'label': 'DMSO'},
              'Y27_1':{'color': '#ffff1a','marker':'o', 'label': '1µM Y27'},
               'none':{'color': '#2986CC','marker':'o', 'label': 'C-OptoRhoA\n(No Light)'},
               }

celltypes = ['optoRhoA']
drugs = ['blebbi_10', 'dmso_10', 'Y27_1', 'none'] #, 'none', 'activation']
magField = [15.0]
activationfreq = [0, 3]
firstAct = [-1, 1]
dates = ['23-03-24', '23-02-02']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            # (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 10),
            # (data['normal field'].apply(lambda x : x in magField)),
            (data['compression duration'] == '1.5s'),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['drug'].apply(lambda x : x in drugs)),
            # (data['activation frequency'].apply(lambda x : x in activationfreq)),
            # (data['first activation'].apply(lambda x : x in firstAct)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs

styleDf = pd.DataFrame(styleDict)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleDict)

# pairs = [['Y27_10', 'none'], ['blebbi_10', 'none'], ['none', 'activation']]
pairs = [['blebbi_10', 'dmso_10']]

plotChars = {'color' : '#000000', 'fontsize' : 15}
plotTicks = {'color' : '#000000', 'fontsize' : 12}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

#%%% Cell-dependent NLR Variability

plotter = df

plotter = plotter.groupby('cellID', \
                          group_keys=False).apply(pf.NLR_normalize_by_first_six,
                                                  'NLI_mod', clip = (0, 6),
                                                  col = 'compNum')


plotter2 = avgDf[('cellID', 'first')]
colorblind_type = "Deuteranomaly"

unique_cellIDs = plotter['cellID'].unique()
Ncells = len(unique_cellIDs)
palette = distinctipy.get_colors(Ncells, colorblind_type=colorblind_type)

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))
cellID_to_color = {cell: palette[i] for i, cell in enumerate(unique_cellIDs)}

plottingParams = {'data':plotter, 
                  'x' : ('cellID'), 
                  'y' : ('NLI_mod_norm'),
                  'hue':'cellID',
                  'palette':cellID_to_color,
                  's':4
                 }

sns.swarmplot(**plottingParams, ax = ax)
for collection in ax.collections:
    collection.set_edgecolor('k')
    collection.set_linewidth(0.5) 


avg_var = plotter['NLI_mod_std'].mean()

for _, row in plotter.iterrows():
    plt.errorbar(x=row['cellID'], y=row['NLI_mod_ref'], yerr=row['NLI_mod_std'],
                 fmt="none", color=cellID_to_color[row['cellID']],
                 capsize=5, alpha=0.4, lw = 0.3,  capthick=1)
    
    
text = 'Average STD of Cells = {:.2f}'.format(avg_var)

plt.text(0, 3, text, fontsize=12)
x_labels = ['C'+str(i+1) for i in range(Ncells)]
ax.set_xticks(ax.get_xticks(), labels =x_labels, rotation = 90, **plotTicks)
plt.ylim(-3, 3.5)
# ax.get_legend().remove()
plt.savefig((dirToSave + '{:}_NLI_StandardDev.pdf').format(condCat), dpi = 200)

#%%% Cell-dependent NLR evolution

plotter = avgDf

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))

plottingParams = {'data':plotter, 
                  'x' : ('drug', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'hue':('cellID','first'),
                  'palette':cellID_to_color,
                  's':10,
                 
                 }

sns.swarmplot(**plottingParams, ax = ax)
plt.ylim(-3, 3.5)

var_pop = plotter[('NLI_mod', 'mean')].var()
text = 'STD of the Population = {:.2f}'.format(var_pop)
plt.text(0, 3, text, fontsize=12)
ax.get_legend().remove()
plt.savefig((dirToSave + '{:}_NLI_StdDevofMean.pdf').format(condCat), dpi = 200)


#%%% Plotnine jitter plots

#%%%% NLI
measure = 'NLI_mod'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1,2))


plt.xticks([1,2,3,4], labels,fontsize=15, color = '#000000')
plt.yticks(**plotTicks)
plt.yticks(**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + 'C-OptoRhoA_Drugs-DoseResponse-Blebbi.pdf'))

#%%%% H0

measure = 'bestH0_log'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure, logScale = True,
                     pairs = pairs, stat = stat, palette = palette_cond, pointSize = 2.5,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2,3,4], labels,fontsize=15, color = '#000000')
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)

plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + 'C-OptoRhoA_Drugs-DoseResponse-Blebbi.pdf'))


#%%%% E_effective

measure = 'E_eff_log'
stat = 'mean'
plot= pf.plotnine_jitter(avgDf, condCol = condCol, condCat = condCat, measure = measure,  pointSize = 2.5,
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2,3,4], labels,fontsize=15, color = '#000000')
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + measure + 'C-OptoRhoA_Drugs-DoseResponse-Blebbi.pdf'))

#%% Calling data - Crosslinker UtCH-Cry2

"""
Task = '24-05-29 & 24-09-05 & 24-02-21 & 24-09-24 & 24-09-12'
        
"""

filename = 'VWC-Chadwick_Chapter-2_UtCH-Cry2_25-04-01'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/UtCH_Replots/'
data = pf.createDataTable(GlobalTable, fitsSubDir = filename)
plt.style.use('seaborn-v0_8')


#%%% Filters For Y27 Dose Response

styleDict =  {#'none':{'color': '#999999','marker':'o', 'label': 'Non-Induced'},
              'doxy':{'color': '#FFA500','marker':'o', 'label': 'No Light'},
              'doxy_act':{'color': '#664200','marker':'o', 'label': '+Crosslink.'},
              #'Y27_10':{'color': '#bfbf7f','marker':'o', 'label': 'Non-Induced\n+Y27'},
               'doxy_2_Y27_10':{'color': '#808000','marker':'o', 'label': 'Y27\nNo Light'},
               'doxy_2_Y27_10_act':{'color': '#333300','marker':'o', 'label': 'Y27\n+Crosslink.'},
               }


option ='doxy_act'
if option == 'doxy_act':
    drugs = ['doxy', 'doxy_act']
    
elif option == 'doxy_2_Y27_10_act':
    drugs = ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']


celltypes = ['uth-cry2']
# drugs = ['doxy', 'doxy_act']
# drugs = ['none', 'doxy', 'doxy_act', 'Y27_10', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
# drugs = [  'doxy_2_Y27_10', 'doxy_2_Y27_10_act']

# drugs = [ 'doxy', 'doxy_act', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']

magField = [5.0]
activationfreq = [0, 1]
firstAct = [-1, 1]
dates = ['24-05-29',  '24-09-05' ,'24-09-24', '24-09-12']

# dates = ['24-09-05' ,'24-09-24', '24-09-12']

excludedManips = ['24-09-05_M5', '24-09-05_M6', '24-09-05_M7'] #Light Coditions too high

Filters = [
            (data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            # (data['compNum'] <= 7),
            ~(data['manipId'].apply(lambda x : x in excludedManips)),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
# condCol, condCat = 'drug', drugs
condCol, condCat = 'drug', drugs
df = pf.NLIcorr(df)

# comps = np.linspace(1, 10, 10)
# condCol, condCat = 'compNum', comps

styleDf = pd.DataFrame(styleDict)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleDict)

# pairs = [['none', 'doxy'], ['doxy', 'doxy_act'], ['Y27_10', 'doxy_2_Y27_10'],  ['none', 'Y27_10'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']]
pairs = [['doxy', 'doxy_act']] #, ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']]

# pairs = [['24-05-29', '24-02-21'], ['24-02-21', '24-09-05'], ['24-05-29', '24-09-24'], ['24-02-21', '24-09-12']]


plotChars = {'color' : '#000000', 'fontsize' : 12}
plotTicks = {'color' : '#000000', 'fontsize' : 12}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

#%%% NLR, Displot


disData = avgDf.reset_index()

disData.columns = [
    '_'.join(map(str, col)).strip() if isinstance(col, tuple) else str(col)
    for col in disData.columns
]

plottingParams = {
    'data': disData,
    'col': f'{condCol}_first',  
    'y': 'NLI_mod_mean',      
    'stat':'percent',
    'bins':15,
    'hue':f'{condCol}_first',  
    'palette': palette_cond,
    'line_kws':{"linewidth": 3},
    'kde':True,
    }

# fig, ax = plt.subplots(figsize=(15/SCALE_px_cm, 15/SCALE_px_cm))

ax = sns.displot( **plottingParams)

ax.tick_params(axis='both', labelsize=plotTicks['fontsize'])


plt.ylabel('Mean NLR per cell', fontweight='bold', **plotChars)

plt.tight_layout()
# fig.title('Mean NLR per cell', fontweight='bold', **plotChars)


# plt.savefig((dirToSave + '{:}_Displot.pdf'.format(option)).format( str(condCat)), dpi = 200)
plt.show()

#%%% NLR, Displot per compression

plottingParams = {
    'data': df,
    'col': condCol,  
    'y': 'NLI_mod',      
    'stat':'percent',
    'bins':15,
    'hue':condCol,  
    'palette': palette_cond,
    'line_kws':{"linewidth": 3},
    'kde':True,
    }

fig, ax = plt.subplots(figsize=(15/SCALE_px_cm, 15/SCALE_px_cm))

g = sns.displot( **plottingParams)

ax.tick_params(axis='both', labelsize=15)
for ax in g.axes.flat:
    ax.axhline(y=0, ls='--', color='red', lw=2, alpha=0.5)
    ax.tick_params(axis='both', labelsize=15)

plt.ylabel('NLR per compression', fontweight='bold', **plotChars)

plt.tight_layout()

plt.savefig((dirToSave + '{:}_Displot_percompression.pdf'.format(option)).format( str(condCat)), dpi = 200)
plt.show()

#%%% NLI vs. Compression

avgDf = pf.createAvgDf(df, condCol)
mask = (
    ((avgDf[(condCol, 'first')] == condCat[1]) & (avgDf[('compNum', 'count')] > 8)) |
    ((avgDf[(condCol, 'first')] == condCat[0]) & (avgDf[('compNum', 'count')] > 5))
    )

avgDf = avgDf.loc[mask]

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

cellsChosen = dfPairs[('dateCell', 'first')].dropna().unique()

random_10 = (pd.Series(cellsChosen).sample(8)).values

toPlot = df[df['dateCell'].apply(lambda x : x in random_10)]

toPlot['compNum'][toPlot[condCol] == option] = toPlot['compNum'] + 6

colorblind_type = "Deuteranomaly"
palette = distinctipy.get_colors(len(toPlot['dateCell'].unique()), colorblind_type=colorblind_type)

y = 'NLI_mod'

plottingParams1 = {'data':toPlot,
                  'x':'compNum', 
                  'y':y, 
                  'hue':'dateCell',
                  'marker':'o',
                  'palette':palette,
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot,
                  'x':'compNum', 
                  'y':y, 
                  'marker':'o',
                  'color':'black'
                 }


fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)
# ax = sns.lineplot(**plottingParams2)

for i in range(1):
    plt.axvline(x= 6.1, color='blue', linestyle='-', 
                ymin=0, ymax=0.15, linewidth=4)
    
N = 2  # You can change N to any value
for i in range(0, 10):
    plt.axvline(x= 7.1 + (i*1), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
    
    
# ax.get_legend().remove()
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
plt.ylim(-3, 3)
plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('NLR', fontweight='bold', **plotChars)

plt.show()
plt.savefig(os.path.join(dirToSave, '{:}vComp_R298_Comp5_UtCH_{:}.pdf'.format(y, option)), dpi = 200)

#%%%% H0


y = 'bestH0_log'

plottingParams1 = {'data':toPlot,
                  'x':'compNum', 
                  'y':y, 
                  'hue':'dateCell',
                  'marker':'o',
                  'palette':palette,
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot,
                  'x':'compNum', 
                  'y':y, 
                  'marker':'o',
                  'color':'black'
                 }


fig, ax = plt.subplots(figsize = (14/SCALE_px_cm,12/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)
ax = sns.lineplot(**plottingParams2)

for i in range(1):
    plt.axvline(x= 6.1, color='blue', linestyle='-', 
                ymin=0, ymax=0.15, linewidth=4)
    
N = 2  # You can change N to any value
for i in range(0, 10):
    plt.axvline(x= 7.1 + (i*1), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
    
ax.set_yscale('log')
ax.get_legend().remove()

y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
plt.yticks(**plotTicks)

# plt.ylim(0, 1200)
plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Cortical Thickness (nm)', fontweight='bold', **plotChars)

plt.show()
plt.savefig(os.path.join(dirToSave, '{:}vComp_R298_Comp5_UtCH_{:}.pdf'.format(y, option)), dpi = 200)


#%%%% E_eff
y = 'E_eff_log'

plottingParams1 = {'data':toPlot,
                  'x':'compNum', 
                  'y':y, 
                  'hue':'dateCell',
                  'marker':'o',
                  'palette':palette,
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot,
                  'x':'compNum', 
                  'y':y, 
                  'marker':'o',
                  'color':'black'
                 }


fig, ax = plt.subplots(figsize = (14/SCALE_px_cm,12/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)
ax = sns.lineplot(**plottingParams2)

for i in range(1):
    plt.axvline(x= 6.1, color='blue', linestyle='-', 
                ymin=0, ymax=0.15, linewidth=4)
    
N = 2  # You can change N to any value
for i in range(0, 10):
    plt.axvline(x= 7.1 + (i*1), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
    
    
ax.get_legend().remove()
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
# plt.ylim(0, 1200)
plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Mean NLR per cell (nm)', fontweight='bold', **plotChars)

plt.show()
plt.savefig(os.path.join(dirToSave, '{:}vComp_R298_Comp5_UtCH_{:}.pdf'.format(y, option)), dpi = 200)



#%%% NLI - Rainplot 

y = 'bestH0'
plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : y,
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(12/SCALE_px_cm, 10/SCALE_px_cm))

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond,# labels = labels,
                              pairs = pairs, shiftBox = 0.15, shiftSwarm = -0.07,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 10,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,4.5)
plt.xticks(color = 'black', fontsize = 13)
# plt.title('NLR per Compression\n'+str(dates), fontweight='bold', **plotChars)
plt.yticks(**plotTicks)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_gLobalActivation_NLRrainplot.pdf'.format(str(dates), y)), dpi = 200)
plt.show()

#%%% E vs H0

toPlot = df[df['dateCell'].apply(lambda x : x in pairedCells)]
dfPairs, pairedCells = pf.dfCellPairs(avgDf)

# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax, dataSlopes = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,
#                                               hueType = condCol, h_ref = 500,
#                                               colorScheme = 'white', palette = palette_cond)
# fig.suptitle(str(dates), **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(figsize=(18/SCALE_px_cm, 10/SCALE_px_cm))
fig, axes, avgDf_EvH = pf.EvH0_LogCellAvg(fig, axes,  dfPairs, condCat, condCol, hueType = condCol,
                                      colorScheme = 'white', palette = palette_cond)
# fig.suptitle(str(dates), **plotChars)

plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.pdf').format(str(dates), str(condCat)))

avgDf =  pf.createAvgDf(dataSlopes, condCol, dataFluoPath = None, e_norm = True)


#%%% Plotnine paired plots
#%%%% NLI - Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)



plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     figsize = (15/SCALE_px_cm,10/SCALE_px_cm),
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'less',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1,2))


plt.xticks([1,2,3,4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)


plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_Paired.pdf'.format(measure, str(dates))))

#%%%% NLR - Normalized Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                          figsize = (15/SCALE_px_cm,10/SCALE_px_cm),
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2, 3, 4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.axhline(y= 0, ls = '--', color = 'red', lw = 1, alpha = 0.5)

plt.xlabel('')
plt.title('Normalized NLR per Cell', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_normPaired.pdf'.format(measure, str(dates))))

#%%%% H0- Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     figsize = (15/SCALE_px_cm,10/SCALE_px_cm),
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2,3,4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_Paired.pdf'.format(measure, str(dates))))

#%%%% BestH0 - Normalized Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot, dfPairsPlot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, figsize = (15/SCALE_px_cm,10/SCALE_px_cm),
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0,2.5))


plt.xticks([1,2, 3, 4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Thickness per Cell', fontweight='bold', **plotChars)
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_normPaired.pdf'.format(measure, str(dates))))

plt.show()


#%%%% E_eff - Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     figsize = (15/SCALE_px_cm,10/SCALE_px_cm),
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2,3,4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity per cell (kPa)', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_Paired.pdf'.format(measure, str(dates))))

#%%%% E_eff - Normalized Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot, dfPairsPlot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'two-sided',
                     figsize = (15/SCALE_px_cm,10/SCALE_px_cm),
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 4))


plt.xticks([1,2, 3, 4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired.pdf'))

plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_Paired.pdf'.format(measure, str(dates))))


#%%% Plotnine paired plots - normalized
#%%%% NLI - Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)



plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2,3,4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_NormPaired.pdf'.format(measure, str(dates))))

#%%%% H0- Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                          figsize = (15/SCALE_px_cm,10/SCALE_px_cm),
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2))


plt.xticks([1,2,3,4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_NormPaired.pdf'.format(measure, str(dates))))

#%%%% E_eff - Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure,
                          figsize = (15/SCALE_px_cm,10/SCALE_px_cm),
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0,3))


plt.xticks([1,2,3,4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity per cell (kPa)', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2_Drugs-Activation_NormPaired.pdf'.format(measure, str(dates))))


#%%% VWC %F_100

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'bestH0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =(y_labels),**plotTicks)

# plt.xlim(-0.5, 2.5)

plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness (nm)', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2.pdf'.format(y, str(dates))))
#%%% E_effective

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'E_eff_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.5,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (8/SCALE_px_cm,8/SCALE_px_cm), tight_layout = True)
plt.yscale('log')
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =(y_labels)/1000,**plotTicks)


plt.ylabel('')
plt.xlabel('')
plt.title('Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + '{:}-{:}_UtCH-Cry2.pdf'.format(y, str(dates))))
