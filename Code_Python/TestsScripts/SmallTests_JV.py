# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:37:51 2021

@author: JosephVermeil
"""
# %% statannotations 1/2

import numpy as np
import pandas as pd
import seaborn as sns
import skimage as skm
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

from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest


# %%

A = np.array([1, 3, 6, 2, 10, 9, 4, 5])
print(A)
print(np.sort(A))
print(np.sort(A)[2:-3])

# %%

N = 20

# P = sns.color_palette("hls", N)
P = sns.color_palette("husl", N)
P_cmap = matplotlib.colors.ListedColormap(P, 'my_cmap')

print(P_cmap.colors)

fig, ax = plt.subplots(1, 1)
ax.set_prop_cycle(plt.cycler("color", P_cmap.colors))
TT = np.linspace(0, 2*np.pi, 360)
dT = np.linspace(0, 1*np.pi, N)
A = np.zeros((N, 360))
for k in range(len(dT)):
    t = dT[k]
    A[k,:] = np.sin(TT + t)
    ax.plot(TT, A[k,:])


# %% Test colors 

cl = matplotlib.colormaps['Set2'].colors + matplotlib.colormaps['Set1'].colors

N = 20
fig, ax = plt.subplots(1, 1)
ax.set_prop_cycle(plt.cycler("color", cl))
TT = np.linspace(0, 2*np.pi, 360)
dT = np.linspace(0, 1*np.pi, N)
A = np.zeros((N, 360))
for k in range(len(dT)):
    t = dT[k]
    A[k,:] = np.sin(TT + t)
    ax.plot(TT, A[k,:])
    
plt.show()

# %%

V = np.linspace(1, 1000, 1600)

Npp = 50
Ntot = len(V)
Nb = Ntot//Npp
step = 100//Nb
Lp = [p for p in range(0, 100, step)]
bins = np.percentile(V, Lp)


# %%

list_image = ['im   0', 'im   1', 'im   2', 'im   3', 'im 100', 'im 101', 'im 102', 'im 103',
              'im 200', 'im 201', 'im 202', 'im 203', 'im 300', 'im 301', 'im 302', 'im 303',]

array_images = np.array(list_image)
array_images_2 = np.sort(array_images)


# %%

list_image = ['C1', 'C2', 'C10', 'C11', 'C20', 'C21']

array_images = np.array(list_image)
array_images_2 = np.sort(array_images)

print(array_images)
print(array_images_2)

# %%

# srcDir = 'E:/24-06-14_Chameleon_Compressions/All_tifs/24-06-14_M2_P1_C2-1_disc20um_L501'
srcDir = 'D:/MagneticPincherData/Raw/24.06.14_Chameleon/24-06-14_M2_P1_C2-2_disc20um_L502'

lst_f = os.listdir(srcDir)
allFiles = [os.path.join(srcDir, f) for f in lst_f]

ic = skm.io.ImageCollection(allFiles, conserve_memory = True)
stack = skm.io.concatenate_images(ic)

# %%

ic.files.sort()

# %%

print(ic.files)

# %%

fig, ax = plt.subplots(1, 1)

# ax.plot([1, 2], [1, 2], ls='=')

# %% Sort

A = [5, 1, 2, 4, 3]
B = [1, 2, 3, 4, 5]

x = np.array([A, B]).T

y = x[x[:, 0].argsort()]

print(x)
print(y)


# %% Make getPixelsInside(I, Y, X) function

    
def getPixelsInside(I, Y, X):
    nx, ny = I.shape[1], I.shape[0]
    contour = np.array([Y, X]).T
    path_edge = mpltPath.Path(contour)
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    pp = np.array([[x, y] for x, y in zip(xx.flatten(), yy.flatten())])
    inside = path_edge.contains_points(pp).reshape(ny, nx)
    return(inside)






# %% Transform test

import skimage as skm

img = skm.data.astronaut()
img = skm.color.rgb2gray(img[40:50, 40:50])

fig, axes = plt.subplots(1, 2, figsize=(12,6))

ax = axes[0]
ax.imshow(img, vmin=0.04, vmax=0.08)

tform = skm.transform.EuclideanTransform(translation=(-1, 0))
img_tformed = skm.transform.warp(img, tform)


ax = axes[1]
ax.imshow(img_tformed, vmin=0.04, vmax=0.08)

plt.show()

# %% TEST METADATA

path = "D://MagneticPincherData//Raw//24.02.27_Chameleon//Clean//Test//3T3-LifeActGFP_PincherFluo_C7_off4um-01.czi"
dstXmlPath = "D://MagneticPincherData//Raw//24.02.27_Chameleon//Clean//Test//3T3-LifeActGFP_PincherFluo_C7_off4um-01_meta.txt"

# with open(path, errors="ignore") as f:
#     metadata = str(f.readlines())
#     keyword = "<Plane"
#     searchedLines = re.findall(keyword, metadata)
#     for L in searchedLines:
#         print(L)
    
    
# from aicsimageio import AICSImage

# # Get an AICSImage object
# img = AICSImage(path)
# data = img.data  # returns 6D STCZYX numpy array
# dims = img.dims  # returns string "STCZYX"
# shape = img.shape  # returns tuple of dimension sizes in STCZYX order


from aicsimageio.readers import CziReader
from ome_types import to_xml
import xml.etree.ElementTree as ET

# Get reader
reader = CziReader(path, include_subblock_metadata=True)
data = reader.data  # returns all image data from the file
dims = reader.dims  # reads the dimensions found in metadata
shape = reader.shape  # returns the tuple of dimension sizes in same order as dims
ome_meta = reader.ome_metadata

xa = reader.xarray_data


# print(reader.channel_names)  # returns a list of string channel names found in the metadata
# print(reader.physical_pixel_sizes.Y)  # returns the Y dimension pixel size as found in the metadata
# print(reader.physical_pixel_sizes.X)  # returns the X dimension pixel size as found in the metadata

xml_ome_meta = to_xml(ome_meta)

# tree = ET.ElementTree(meta)
# ET.indent(tree, space="\t", level=0)
# tree.write(dstXmlPath, encoding="utf-8")

# from aicsimageio.readers import BioformatsReader

# reader = BioformatsReader(path)

ome_meta.getattr('Experimenter')

# %%

from pylibCZIrw import czi as pyczi

path = "D://MagneticPincherData//Raw//24.02.27_Chameleon//Clean//Test//3T3-LifeActGFP_PincherFluo_C7_off4um-01.czi"

# open the CZI document to read the
with pyczi.open_czi(path) as czidoc:
    # define some plane coordinates
    plane_1 = {'C': 0, 'Z': 2, 'T': 1}
    plane_2 = {'C': 1, 'Z': 3, 'T': 2}

    # equivalent to reading {'C': 0, 'Z': 0, 'T': 0}
    frame_0 = czidoc.read()

    # get the shape of the 2d plane - the last dime indicates the pixel type
    # 3 = BGR and 1 = Gray
    print("Array Shape: ", frame_0.shape)

    # get specific planes 
    frame_1 = czidoc.read(plane=plane_1)
    frame_2 = czidoc.read(plane=plane_2)


# %%

from statsmodels.robust.norms import HuberT

H = HuberT(t=6)
# H1 = HuberT(t=1)
# H2 = HuberT(t=2)

X = np.linspace(-10, 10, 1000)
Y = H.rho(X)
d = H.psi_deriv(X)

plt.plot(X, Y)
plt.plot(X, d)
plt.show()



# %% Test image translation
import scipy.ndimage as ndi
import UtilityFunctions as ufun
from skimage import io, filters, exposure, measure, transform, util, color
import matplotlib.patches as patches


def gaussian2D(size, sigma=1, center=(0, 0)):
    muu=0
    x0, y0 = center
    x, y = np.meshgrid(np.linspace(-1, 1, size),
                       np.linspace(-1, 1, size))
    dst = np.sqrt((x-x0)**2+(y-y0)**2)
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    return(gauss)
 
imSize=15
cleanSize = 9
cleanCenter = (cleanSize) // 2
roughSize = 11
roughCenter = (roughSize) // 2

Iraw = gaussian2D(imSize, sigma=0.4, center = (0.12, -0.22))
Y, X = ndi.center_of_mass(Iraw)
xb1, yb1, xb2, yb2, validROI = ufun.getROI(roughSize, X, Y, 1000, 1000)

fig1, ax1 = plt.subplots(1, 1)
ax = ax1
ax.imshow(Iraw)
ax.plot(imSize//2, imSize//2, 'b+')
ax.plot(X, Y, 'r+')
ax.plot(np.round(X), np.round(Y), 'g+')
ax.axvline(xb1, c='r', ls='--')
ax.axvline(xb2-1, c='r', ls='--')
ax.axhline(yb1, c='r', ls='--')
ax.axhline(yb2-1, c='r', ls='--')
print(X, Y)
print(np.round(X), np.round(Y))
print(xb1, yb1, xb2, yb2, validROI)

I_roughRoi = Iraw[yb1:yb2, xb1:xb2]
fig2, ax2 = plt.subplots(1, 1)
ax = ax2
ax.imshow(I_roughRoi)
ax.plot(roughSize//2, roughSize//2, 'g+')
ax.plot(X-xb1, Y-yb1, 'r+')

xc1, yc1 = X-xb1, Y-yb1
translation = (xc1-roughCenter, yc1-roughCenter)
tform = transform.EuclideanTransform(rotation=0, translation = translation)
F_tmp = transform.warp(I_roughRoi, tform, order = 1, preserve_range = True)
fig3, ax3 = plt.subplots(1, 1)
ax = ax3
ax.imshow(F_tmp)
ax.axvline(roughCenter, c='r', ls='--')

I_cleanRoi = np.copy(F_tmp[roughCenter-cleanSize//2:roughCenter+cleanSize//2+1,\
                            roughCenter-cleanSize//2:roughCenter+cleanSize//2+1])
Y2, X2 = ndi.center_of_mass(I_cleanRoi)
cleanCenter = (cleanSize) // 2
fig4, ax4 = plt.subplots(1, 1)
ax = ax4
ax.imshow(I_cleanRoi)
ax.plot(X2, Y2, 'r+')
ax.axvline(cleanCenter, c='r', ls='--')

plt.show()

#### part2

imSize=15
cleanSize = 9
cleanCenter = (cleanSize) // 2
roughSize = 11
roughCenter = (roughSize) // 2

Iraw = gaussian2D(imSize, sigma=0.4, center = (0.12, -0.22))
Y, X = ndi.center_of_mass(Iraw)
xb1, yb1, xb2, yb2, validROI = ufun.getROI(roughSize, X, Y, 1000, 1000)

I_roughRoi = Iraw[yb1:yb2, xb1:xb2]


xc1, yc1 = X-xb1, Y-yb1
translation = (xc1-roughCenter, yc1-roughCenter)
tform = transform.EuclideanTransform(rotation=0, translation = translation)
F_tmp = transform.warp(I_cleanRoi, tform, order = 1, preserve_range = True)

I_cleanRoi = np.copy(F_tmp[roughCenter-cleanSize//2:roughCenter+cleanSize//2+1,\
                            roughCenter-cleanSize//2:roughCenter+cleanSize//2+1])


# %% Test shapely

# %%% Plot Polygon

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon, plot_points

# from figures import GRAY, RED, SIZE, set_limits

fig = plt.figure(1, dpi=90)

# 3: invalid polygon, ring touch along a line
ax = fig.add_subplot(121)

exte = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
inte = [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1), (0.5, 0)]
polygon = Polygon(exte, [inte])

plot_polygon(polygon, ax=ax, add_points=False, color='red')
plot_points(polygon, ax=ax, color='gray', alpha=0.7)

ax.set_title('c) invalid')


#4: invalid self-touching ring
ax = fig.add_subplot(122)
ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
int_1 = [(0.5, 0.25), (1.5, 0.25), (1.5, 1.25), (0.5, 1.25), (0.5, 0.25)]
int_2 = [(0.5, 1.25), (1, 1.25), (1, 1.75), (0.5, 1.75)]
polygon = Polygon(ext, [int_1, int_2])

plot_polygon(polygon, ax=ax, add_points=False, color='red')
plot_points(polygon, ax=ax, color='gray', alpha=0.7)

ax.set_title('d) invalid')


plt.show()

# %%% Find PIA & radius

from shapely.ops import polylabel
from shapely import Polygon, LineString, get_exterior_ring, distance
from shapely.plotting import plot_polygon, plot_points, plot_line

fig = plt.figure(1, dpi=90)

line = LineString([(0, 0), (50, 200), (100, 100), (20, 50), (-100, -20), (-100, +200)]) #.buffer(100)
polygon = line.buffer(100)

ax = fig.add_subplot(121)
plot_line(line, ax=ax, add_points=False, color='blue')
plot_points(line, ax=ax, color='gray', alpha=0.7)
plot_polygon(polygon, ax=ax, add_points=False, color='red')
# plot_points(polygon, ax=ax, color='gray', alpha=0.7)

label = polylabel(polygon, tolerance=1)
ExtR = get_exterior_ring(polygon)
D = distance(label, ExtR)
circle = label.buffer(D)


ax = fig.add_subplot(122)

plot_line(ExtR, ax=ax, add_points=False, color='blue')
# plot_points(line, ax=ax, color='gray', alpha=0.7)
plot_polygon(circle, ax=ax, add_points=False, color='red')
plot_points(label, ax=ax, color='green', alpha=1)


# %% Test colors


A = np.arange(1,13)
B = A.reshape((4,3))
C = np.sum(B, axis = 1)
C



# %% Test colors
N = 6
fig, ax = plt.subplots(1, 1)
ax.set_prop_cycle(plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N))))
TT = np.linspace(0, 2*np.pi, 360)
dT = np.linspace(0, 1*np.pi, 6)
A = np.zeros((6, 360))
for k in range(len(dT)):
    t = dT[k]
    A[k,:] = np.sin(TT + t)
    ax.plot(TT, A[k,:])
    
plt.show()

# %% Test sorting

A = np.array([3.2, 1.6, 6.4, 0.8])
r = np.argsort(A)
B = list(A[r])


# %% Test sorting

import numpy as np
import pandas as pd

d = {'grade':[13, 16, 18, 12, 6, 7, 15, 14, 4, 18, 4, 13], 
     'coeff':[2, 1, 5, 4, 1, 7, 3, 2, 5, 4, 9, 8], 
     'course':['m', 'p', 'l', 'm', 'm', 'p', 'p', 'l','l','l','l', 'l']}

df = pd.DataFrame(d)

order = ['l', 'm', 'p']
dict_order = {order[i]:i for i in range(len(order))}
df['course_order'] = df['course'].apply(lambda x : dict_order[x])

df_sorted = df.sort_values(by='course_order', axis=0)


# %% Test new interp function

from skimage import io

import UtilityFunctions as ufun

imagePath = 'C://Users//JosephVermeil//Desktop//cycle.png'

I = io.imread(imagePath)

fig1, ax1 = plt.subplots(1,1)
ax1.imshow(I, cmap='gray')

I2 = ufun.resize_2Dinterp(I, new_nx=200, new_ny=200, fx=None, fy=None)

fig2, ax2 = plt.subplots(1,1)
ax2.imshow(I2, cmap='gray')

plt.show()


#%% 2 X-axis

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

X = np.linspace(0,0.2,1000)
Y = np.cos(X/10)

ax1.plot(X,Y)
ax1.set_xlabel(r"Original x-axis: $X$")

def tick_function(eps, h0):
    H = h0 - 3*h0*eps
    return ["%.1f" % h for h in H]

h0 = 500

ax1.set_xlim(ax1.get_xlim())

ax2.set_xlim(ax1.get_xlim())
new_tick_locations = np.array(ax1.get_xticks()) # [0, 0.05, 0.1, 0.15, 0.2]
print(new_tick_locations)
ax1.set_xticks(new_tick_locations)
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations, h0))
ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")
# ax1.set_xlim([0, 0.3])
ax2.grid()
plt.show()

# %% sth
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

X = np.linspace(0, 50, 10000)
Y = 2 + 10/(1 + np.exp(-0.2*(X-30)))

plt.plot(X,Y)
plt.show()

# %% idxAnalysis

import numpy as np
import pandas as pd
import UtilityFunctions as ufun

A = np.array([0, 1, 2, -1, -2, 3, -5, 2, 4, 1])
df = pd.DataFrame({'A':A})
vals = df['A'].unique()

N = np.max(df['A'])
idx = [i for i in range(N+1) if i in vals]   


# %% regions

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import UtilityFunctions as ufun

A = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1])
A2 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])
B, n = ndi.label(A)

C = np.logical_xor(A2, A)



# %% sth

import numpy as np
import pandas as pd
import UtilityFunctions as ufun

def fun(tsDf, i, task):
    Npts = len(tsDf['idxAnalysis'].values)
    iStart = ufun.findFirst(np.abs(tsDf['idxAnalysis']), i+1)
    iStop = iStart + np.sum((np.abs(tsDf['idxAnalysis']) == i+1))
    
    if task == 'compression':
        mask = (tsDf['idxAnalysis'] == i+1).values
    elif task == 'precompression':
        mask = (tsDf['idxAnalysis'] == -(i+1)).values
    elif task == 'compression & precompression':
        mask = (np.abs(tsDf['idxAnalysis']) == i+1).values
    elif task == 'previous':
        i2 = ufun.findFirst_V2(i+1, np.abs(tsDf['idxAnalysis']))
        if i2 == -1:
            i2 = tsDf['idxAnalysis'].size
        i1 = Npts - ufun.findFirst_V2(i, np.abs(tsDf['idxAnalysis'][::-1]))
        if i == 0:
            i1 = 0
        mask = np.array([((j >= i1) and (j < i2)) for j in range(Npts)])
    elif task == 'following':
        i2 = ufun.findFirst_V2(i+2, np.abs(tsDf['idxAnalysis']))
        if i2 == -1:
            i2 = tsDf['idxAnalysis'].size
        i1 = Npts - ufun.findFirst_V2(i+1, np.abs(tsDf['idxAnalysis'][::-1]))
        if i1 > Npts:
            i1 = 0
        mask = np.array([((j >= i1) and (j < i2)) for j in range(Npts)])
    elif task == 'surrounding':
        mask = ((np.abs(tsDf['idxLoop']) == i+1) & (np.abs(tsDf['idxAnalysis']) == 0)).values
    
    return(mask)

a1 = np.array([1,1,1,1,1,1,1,1,
               2,2,2,2,2,2,2,2,
               3,3,3,3,3,3,3,3])

a2 = np.array([0,0,-1,-1,1,1,0,0,
               0,0,-2,-2,2,2,0,0,
               0,0,-3,-3,3,3,0,0])

tsDf = pd.DataFrame({'idxLoop' : a1, 'idxAnalysis' : a2})

mask = fun(tsDf, 2, 'compression')

# %% merge


d1 = {'i1':[1,2,3,4,5,6,7,8,9], 
     'i2':[0,1,2,3,4,5,6,7,8], 
     'i3':[11,12,13,14,15,16,17,18,19]}

df1 = pd.DataFrame(d1)

# d2 = [1,2,3,6,7,8,9]
# df2 = pd.Series(d2, name = 'i1')

d2 = {'i1':[1,2,3,6,7,8,9]}
df2 = pd.DataFrame(d2)

df3 = pd.merge(left=df1, right=df2, how='right', on='i1')

# Filter1 = (df['course'] == 'l')
# Filter2 = (df['grade'] > 10)

# gF = Filter1 & Filter2



# %% bool ops


d = {'grade':[13, 16, 18, 12, 6, 7, 15, 14, 20, 18, 19, 13], 
     'coeff':[2, 1, 5, 4, 1, 7, 3, 2, 5, 4, 9, 8], 
     'course':['m', 'p', 'l', 'm', 'm', 'p', 'p', 'l','l','l','l', 'l']}

df = pd.DataFrame(d)

# Filter1 = (df['course'] == 'l')
# Filter2 = (df['grade'] > 10)

# gF = Filter1 & Filter2

df.at[(df['grade'] == 20), 'course']


# %% statannotations 2/2


d = {'grade':[13, 16, 18, 12, 6, 7, 15, 14, 20, 18, 19, 13], 
     'coeff':[2, 1, 5, 4, 1, 7, 3, 2, 5, 4, 9, 8], 
     'course':['m', 'p', 'l', 'm', 'm', 'p', 'p', 'l','l','l','l', 'l']}

df = pd.DataFrame(d)

fig, ax = plt.subplots(1,1)

swarmplot_parameters = {'ax' : ax,
                        'data':    df,
                        'x':       'course',
                        'y':       'grade',
                        'edgecolor'    : 'k', 
                        'linewidth'    : 1,
                        'orient' : 'v'
                        }

# sns.boxplot(data = df, ax=ax,
#             x = 'course', y = 'grade',
#             showfliers=False, color = 'w')

sns.swarmplot(**swarmplot_parameters)

ax.set_ylim([0, 40])
ax.set_xlabel(r"$\bf{This\ is\ bold}$")

# annotator = Annotator(ax, box_pairs, **plotting_parameters)
# annotator.configure(test=test, verbose=verbose).apply_and_annotate()

def addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plotting_parameters):
    listTests = ['t-test_ind', 't-test_welch', 't-test_paired', 
                 'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls', 
                 'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel']
    if test in listTests:
        annotator = Annotator(ax, box_pairs, **plotting_parameters)
        annotator.configure(test=test, verbose=verbose).apply_and_annotate()
    else:
        print(gs.BRIGHTORANGE + 'Dear Madam, dear Sir, i am the eternal god and i command that you have to define this stat test cause it is not in the list !' + gs.NORMAL)
    return(ax)

plt.show()

# %%

d = {'grade':[13, 16, 18, 12, 6, 7, 15, 14, 20, 18, 19, 13], 
     'coeff':[2, 1, 5, 4, 1, 7, 3, 2, 5, 4, 9, 8], 
     'course':['m', 'p', 'l', 'm', 'm', 'l', 'l', 'l_0','l_0','l_0','l', 'm'],
     'iL':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     'idxAnalysis':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

df = pd.DataFrame(d)
indexAction = df[df['course'].apply(lambda x : x.startswith('l'))].index
df.loc[indexAction, 'idxAnalysis'] = df.loc[indexAction, 'iL'] * (-1)
i_startOfSpike = ufun.findFirst('l_0', df['course'].values)
i_endOfSpike = ufun.findLast('l', df['course'].values)
# indexMainPhase = logDf[logDf['Status'].apply(lambda x : x.startswith('Action_main'))].index
df.loc[i_startOfSpike:i_endOfSpike+1, 'idxAnalysis'] *= (-1) # *logDf.loc[i_startOfSpike:i_endOfSpike+1, 'idxAnalysis']

# %%

import seaborn as sns

from statannotations.Annotator import Annotator

df = sns.load_dataset("tips")
x = "day"
y = "total_bill"
order = ['Sun', 'Thur', 'Fri', 'Sat']

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax = sns.boxplot(data=df, x=x, y=y, order=order)

pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")]

annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
annotator.apply_and_annotate()

plt.tight_layout()

# %% Pandas tests

import numpy as np
import pandas as pd
import UtilityFunctions as ufun

d = {'grade':[13, 16, 18, 12, 6, 7, 15, 14, 4, 18, 4, 13], 
     'coeff':[2, 1, 5, 4, 1, 7, 3, 2, 5, 4, 9, 8], 
     'course':['m', 'p', 'l', 'm', 'm', 'p', 'p', 'l','l','l','l', 'l']}

df = pd.DataFrame(d)

valCol = 'grade'
weightCol = 'coeff'
groupCols = ufun.toList('course')
wAvgCol = valCol + '_wAvg'

df['A'] = df[valCol] * df[weightCol]
grouped1 = df.groupby(by=groupCols)
data_agg = grouped1.agg({'A': ['count', 'sum'], 
                         weightCol: 'sum',
                         'grade' : [np.var, np.std, ufun.interDeciles]}).reset_index()

data_agg.columns = ufun.flattenPandasIndex(data_agg.columns)
# data_agg.columns = ['_'.join(col) for col in data_agg.columns.values]

# data_id2 = df.groupby('course').agg('first')

df['id_within_co'] = [np.sum(df.loc[:i,'course'] == df.loc[i,'course']) for i in range(df.shape[0])]

# %% Regex test

import re

joker = r'[\.\d]{2,6}'
s = '0.2_' + joker

text = '0.2_0.0125'

if re.match(s, text):
    print('ok')
    
print(re.match(s, text))

# %% Regex test

import re

template = 'Merci_Monsieur'
s1 = 'Monsieur'
s2 = 'Madame'

if re.search(s1, template):
    res = re.sub(s1, s2, template)
    print(res)
    
print(re.search(s1, template))

# %% Dicts

d1 = {'a':1, 'b':2}

d2 = d1

print(d1, d2)

d2.pop('b')

print(d1, d2)

# %%

# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. Test

def wrapper1(x, **kwargs):
    wrapper2(x, **kwargs)

def wrapper2(x, **kwargs):
    multi, add = 1, 0
    if 'multi' in kwargs:
        multi = kwargs['multi']
    if 'add' in kwargs:
        add = kwargs['add']
    effector(x, multi, add)
    
    

def effector(x, multi = 1, add = 0):
    print(x * multi + add)


wrapper1(2, multi = 34, add = 3)


# %%

# 1. Imports
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import os
import re
import time
import pyautogui
import matplotlib
import traceback
# import cv2

# import scipy
from scipy import interpolate
from scipy import signal

# import skimage
from skimage import io, filters, exposure, measure, transform, util, color
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from datetime import date

# Add the folder to path
COMPUTERNAME = os.environ['COMPUTERNAME']
if COMPUTERNAME == 'ORDI-JOSEPH':
    mainDir = "C://Users//JosephVermeil//Desktop//ActinCortexAnalysis"
    ownCloudDir = "C://Users//JosephVermeil//ownCloud//ActinCortexAnalysis"
    tempPlot = 'C://Users//JosephVermeil//Desktop//TempPlots'
elif COMPUTERNAME == 'LARISA':
    mainDir = "C://Users//Joseph//Desktop//ActinCortexAnalysis"
    ownCloudDir = "C://Users//Joseph//ownCloud//ActinCortexAnalysis"
    tempPlot = 'C://Users//Joseph//Desktop//TempPlots'
elif COMPUTERNAME == 'DESKTOP-K9KOJR2':
    mainDir = "C://Users//anumi//OneDrive//Desktop//ActinCortexAnalysis"
elif COMPUTERNAME =='DATA2JHODR':
    mainDir = "C://Utilisateurs//BioMecaCell//Bureau//ActinCortexAnalysis"
    tempPlot = 'C://Utilisateurs//BioMecaCell//Bureau//TempPlots'



import sys
sys.path.append(mainDir + "//Code_Python")

# from getExperimentalConditions import getExperimentalConditions
import utilityFunctions_JV as jvu


# 2. Pandas settings
pd.set_option('mode.chained_assignment', None)

# 3. Plot settings
# Here we use this mode because displaying images
# in new windows is more convenient for this code.
# %matplotlib qt 
# matplotlib.use('Qt5Agg')
# To switch back to inline display, use : 
# %matplotlib widget or %matplotlib inline
# matplotlib.rcParams.update({'figure.autolayout': True})

SMALLER_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# 4. Other settings
# These regex are used to correct the stupid date conversions done by Excel
dateFormatExcel = re.compile(r'\d{2}/\d{2}/\d{4}')
dateFormatExcel2 = re.compile(r'\d{2}-\d{2}-\d{4}')
dateFormatOk = re.compile(r'\d{2}-\d{2}-\d{2}')

# 5. Global constants
SCALE_100X = 15.8 # pix/µm
NORMAL  = '\033[0m'
RED  = '\033[31m' # red
GREEN = '\033[32m' # green
ORANGE  = '\033[33m' # orange
BLUE  = '\033[36m' # blue


fig, axes = plt.subplots(1, 2, figsize = (8,4))
axes[0].plot(np.arange(0,10,0.5),np.arange(0,100,5)**4)

xtVal = axes[0].get_xticks()
axes[0].set_xticklabels([str(x*4.5) for x in xtVal])

fig.show()

# %%

import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

fig, ax = plt.subplots(1,1)

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="day", y="total_bill", ax = ax,
            hue="smoker", 
            palette=["m", "g"],
            data=tips)
sns.despine(ax = ax, offset=10, trim=True)

plt.show()

# %%

import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1,1)
ax.errorbar([1], [2], [[1.5],[5]], marker = 'o')
plt.show()

# %%

import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dicta={'val' : [1,2,3,4,2.5,3.5], 'cat' : ['A','B','A','A','B','B'], 'sub' : ['q','q','w','w','w','q']}
df = pd.DataFrame(dicta)

S = ['q','q','w','w','w','q']

fig, ax = plt.subplots(1,1)

# sns.stripplot(ax = ax, data = df, x = 'cat', y = 'val')
Y = np.array([[2,3],[4,5]])
sns.stripplot(ax = ax, x = np.array([1,2]), y = Y)

plt.show()

# %%
import  pandas as pd
import numpy as np


class Grandpa:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def m1(self):
        return(self.a>0)

class Dad(Grandpa):
    def __init__(self, a, b):
        super().__init__(a, b)
        
    
    # def m1(self):
    #     return(self.a<0)

# %%
import  pandas as pd
import numpy as np

dicta={'aaa' : [1,2,3,4], 'bbb' : ['p','o','p','p'], 'ccc' : ['q','q','w','w']}
df = pd.DataFrame(dicta)
N = df.loc[df['bbb'] == 'p'].shape[0]
# A = np.array(['youpi' for kk in range(N)])
df.loc[df['bbb'] == 'p', 'ccc'] = 'youpo'


# %%
import  pandas as pd
import numpy as np

dicta={'aaa' : [1,2,3,4], 'bbb' : ['p','o','p','p'], 'ccc' : ['q','q','w','w']}
df = pd.DataFrame(dicta)
N = df.loc[df['bbb'] == 'p'].shape[0]
# A = np.array(['youpi' for kk in range(N)])
df.loc[df['bbb'] == 'p', 'ccc'] = 'youpo'

df.loc[df['bbb'] == 'p', 'aaa'] = df.loc[df['bbb'] == 'p', 'aaa'].apply(lambda x : x*10)

# %%
import numpy as np


def myfun(a):
    a = a + [2]

a = [1]

print(a)

myfun(a)

print(a)


# %% Fastness test
import numpy as np
import time

N = int(1e4)
a = np.zeros((N,N))
b = np.ones((N,N))

T0 = time.time()
# c = (b-a)**2
c =  a.T
T1 = time.time()
# np.square(np.subtract(b,a))
T2 = time.time()

print(T1-T0)
print(T2-T1)

# %% Fastness test
import numpy as np
import time

N = int(2e4)
a = np.zeros((N,N))
b = np.ones((N,N))
A = np.repeat(np.arange(1,N+1), N, axis = 0).reshape(N,N)

T0 = time.time()

m = A / np.mean(A,axis=1)[:,None]

T1 = time.time()

m2 = (A.T / np.mean(A,axis=1)).T

T2 = time.time()

m3 = np.divide(A.T,np.mean(A,axis=1)).T

T3 = time.time()

print(T1-T0)
print(T2-T1)
print(T3-T2)

# %% Fastness test
import numpy as np
import time

def squareDistance(M, V, normalize = False): # MUCH FASTER ! **Michael Scott Voice** VERRRRY GOODE
    """
    Compute a distance between two arrays of the same size, defined as such:
    D = integral of the squared difference between the two arrays.
    It is used to compute the best fit of a slice of a bead profile on the depthograph.
    This function speed is critical for the Z computation process because it is called so many times !
    What made that function faster is the absence of 'for' loops and the use of np.repeat().
    """
    # squareDistance(self.deptho, listProfiles[i], normalize = True)
    # top = time.time()
    # n, m = M.shape[0], M.shape[1]
    # len(V) should be m
    if normalize:
        V = V/np.mean(V)
    MV = np.repeat(np.array([V]), M.shape[0], axis = 0) # Key trick for speed !
    if normalize:
        M = M / np.mean(M,axis=1)[:,None]
    R = np.sum(np.square(np.subtract(M,MV)), axis = 1)
    # print('DistanceCompTime')
    # print(time.time()-top)
    return(R)

N = int(1e4)
a = np.ones((N,N))
b = np.ones(N)*5

T0 = time.time()
R1 = squareDistance(a,b)
T1 = time.time()
R2 = squareDistance(a,b, normalize = True)
T2 = time.time()

T3 = time.time()

print(T1-T0)
print(T2-T1)
print(T3-T2)

# %% Chi2 test

# %%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm

import os
import time
import random
import traceback


from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
# from statannot import add_stat_annotation

pd.set_option('mode.chained_assignment',None)
pd.set_option('display.max_columns', None)

import re
dateFormatExcel = re.compile('\d{2}/\d{2}/\d{4}')
dateFormatOk = re.compile('\d{2}-\d{2}-\d{2}')

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'figure.autolayout': True})

# Add the folder to path
import sys
sys.path.append("C://Users//JosephVermeil//Desktop//ActinCortexAnalysis//Code_Python")
from getExperimentalConditions import getExperimentalConditions
from PincherAnalysis_JV import *

# 5. Global constants
SCALE_100X = 15.8 # pix/µm 
NORMAL  = '\033[0m'
RED  = '\033[31m' # red
GREEN = '\033[32m' # green
ORANGE  = '\033[33m' # orange
BLUE  = '\033[36m' # blue

# %%% Directories

mainDataDir = 'D://MagneticPincherData'
rawDataDir = os.path.join(mainDataDir, 'Raw')
depthoDir = os.path.join(rawDataDir, 'EtalonnageZ')
interDataDir = os.path.join(mainDataDir, 'Intermediate')
figureDir = os.path.join(mainDataDir, 'Figures')
timeSeriesDataDir = "C://Users//JosephVermeil//Desktop//ActinCortexAnalysis//Data_Analysis//TimeSeriesData"
experimentalDataDir = "C://Users//JosephVermeil//Desktop//ActinCortexAnalysis//Data_Experimental"

# %%% Data

expDf = getExperimentalConditions(experimentalDataDir)

cellId = '21-12-08_M1_P2_C1'
df = getCellTimeSeriesData(cellId)



# %%% Functions

def segmentTimeSeries_meca(f, tsDF, expDf):
    #### (0) Import experimental infos
    split_f = f.split('_')
    tsDF.dx, tsDF.dy, tsDF.dz, tsDF.D2, tsDF.D3 = tsDF.dx*1000, tsDF.dy*1000, tsDF.dz*1000, tsDF.D2*1000, tsDF.D3*1000
    thisManipID = split_f[0] + '_' + split_f[1]
    expDf['manipID'] = expDf['date'] + '_' + expDf['manip']
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]

    diameters = str(thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'])
    diameters = diameters.split('_')
    DIAMETER = int(diameters[0])
    EXPTYPE = str(thisExpDf.at[thisExpDf.index.values[0], 'experimentType'])

    Ncomp = max(tsDF['idxAnalysis'])
    
    loopStruct = thisExpDf.at[thisExpDf.index.values[0], 'loop structure'].split('_')
    nUplet = thisExpDf.at[thisExpDf.index.values[0], 'normal field multi images']
    if 'compression' in EXPTYPE:
        loop_totalSize = int(loopStruct[0])
        if len(loopStruct) >= 2:
            loop_rampSize = int(loopStruct[1])
        else:
            loop_rampSize = 0
        if len(loopStruct) >= 3:
            loop_excludedSize = int(loopStruct[2])
        else:
            loop_excludedSize = 0
        loop_ctSize = int((loop_totalSize - (loop_rampSize+loop_excludedSize))/nUplet)


    compField = thisExpDf.at[thisExpDf.index.values[0], 'ramp field'].split('_')
    minCompField = int(compField[0])
    maxCompField = int(compField[1])
    
    hComprList = []
    fComprList = []


    for i in range(1, Ncomp+1): #Ncomp+1):

        #### (1) Segment the compression n°i
        thisCompDf = tsDF.loc[tsDF['idxAnalysis'] == i,:]
        iStart = (findFirst(tsDF['idxAnalysis'], i))
        iStop = iStart+thisCompDf.shape[0]


        listB = thisCompDf.B.values
        
        # Test to check if most of the compression have not been deleted due to bad image quality 
        highBvalues = (listB > (maxCompField + minCompField)/2)
        N_highBvalues = np.sum(highBvalues)
        testHighVal = (N_highBvalues > 20)

        # Test to check if the range of B field is large enough
        minB, maxB = min(listB), max(listB)
        testRangeB = ((maxB-minB) > 0.7*(maxCompField - minCompField))
        thresholdB = (maxB-minB)/50
        thresholdDeltaB = (maxB-minB)/400

        # Is the curve ok to analyse ?
        doThisCompAnalysis = testHighVal and testRangeB # Some criteria can be added here

        if doThisCompAnalysis:
            #### (2) Inside the compression n°i, delimit the compression and relaxation phases

            # Delimit the start of the increase of B (typically the moment when the field decrease from 5 to 3)
            # and the end of its decrease (typically when it goes back from 3 to 5)
            
            try:
                # Correct for bugs in the B data
                if 'compressions' in EXPTYPE:
                    for k in range(1,len(listB)):
                        B = listB[k]
                        if B > 1.25*maxCompField:
                            listB[k] = listB[k-1]

                offsetStart, offsetStop = 0, 0
                minB, maxB = min(listB), max(listB)
                thresholdB = (maxB-minB)/50
                thresholdDeltaB = (maxB-minB)/400 # NEW CONDITION for the beginning of the compression : 
                # remove the first points where the steps in B are very very small

                k = 0
                while (listB[k] < minB+thresholdB) or (listB[k+1]-listB[k] < thresholdDeltaB):
                    offsetStart += int((listB[k] < minB+thresholdB) or ((listB[k+1]-listB[k]) < thresholdDeltaB))
                    k += 1

                k = 0
                while (listB[-1-k] < minB+thresholdB):
                    offsetStop += int(listB[-1-k] < minB+thresholdB)
                    k += 1

                jStart = offsetStart # Beginning of compression
                jMax = np.argmax(thisCompDf.B) # End of compression, beginning of relaxation
                jStop = thisCompDf.shape[0] - offsetStop # End of relaxation
            
            except:
                print(listB)
                print(testRangeB, thresholdB, thresholdDeltaB)

            # Four arrays
            hCompr = (thisCompDf.D3.values[jStart:jMax+1] - DIAMETER)
            fCompr = (thisCompDf.F.values[jStart:jMax+1])

            # Refinement of the compression delimitation.
            # Remove the 1-2 points at the begining where there is just the viscous relaxation of the cortex
            # because of the initial decrease of B and the cortex thickness increases.
            offsetStart2 = 0
            k = 0
            while (k<len(hCompr)-10) and (hCompr[k] < np.max(hCompr[k+1:min(k+10, len(hCompr))])):
                offsetStart2 += 1
                k += 1
            # Better compressions arrays
            hCompr = np.array(hCompr[offsetStart2:])
            fCompr = np.array(fCompr[offsetStart2:])

            hComprList.append(hCompr)
            fComprList.append(fCompr)
            
    return(DIAMETER, hComprList, fComprList)



def compressionFitChadwick_V2(hCompr, fCompr, DIAMETER):
    
    error = False
    
    def chadwickModel(h, E, H0):
        R = DIAMETER/2
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0):
        R = DIAMETER/2
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)

    # some initial parameter values - must be within bounds
    initH0 = max(hCompr) # H0 ~ h_max
    initE = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    
    initialParameters = [initE, initH0]
    
    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, 0)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]


    try:
        params, covM = curve_fit(inversedChadwickModel, fCompr, hCompr, initialParameters, bounds = parameterBounds)

        # Previously I fitted with y=F and x=H, but it didn't work so well cause H(t) isn't monotonous:
        # params, covM = curve_fit(chadwickModel, hCompr, fCompr, initialParameters, bounds = parameterBounds)
        # Fitting with the 'inverse Chadwick model', with y=H and x=F is more convenient

        E, H0 = params
        
        hPredict = inversedChadwickModel(fCompr, E, H0)
        residuals_h = hCompr-hPredict
        
        comprMat = np.array([hCompr, fCompr]).T
        comprMatSorted = comprMat[comprMat[:, 0].argsort()]
        hComprSorted, fComprSorted = comprMatSorted[:, 0], comprMatSorted[:, 1]
        fPredict = chadwickModel(hComprSorted, E, H0)
        residuals_f = fComprSorted-fPredict
        
        deltaCompr = (H0 - hCompr)/1000
        stressCompr = fCompr / (np.pi * (DIAMETER/2000) * deltaCompr)
        strainCompr = deltaCompr / (3*(H0/1000))
        strainPredict = stressCompr / (E*1e6)
        
        # Stats
        alpha = 0.975
        dof = len(fCompr)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = get_R2(hCompr, hPredict)
        #### err
        err = 0.02
        
        # print('fCompr')
        # print(fCompr)
        # print('stressCompr')
        # print(stressCompr)
        # print('strainCompr')
        # print(strainCompr)
        # print('strainPredict')
        # print(strainPredict)

        Chi2 = get_Chi2(strainCompr, strainPredict, dof, err)

        varE = covM[0,0]
        seE = (varE)**0.5
        E, seE = E*1e6, seE*1e6
        confIntE = [E-q*seE, E+q*seE]
        confIntEWidth = 2*q*seE

        varH0 = covM[1,1]
        seH0 = (varH0)**0.5
        confIntH0 = [H0-q*seH0, H0+q*seH0]
        confIntH0Width = 2*q*seH0
        
        
    except Exception:
        print(RED + '')
        traceback.print_exc()
        print('\n')
        error = True
        E, H0, hPredict, R2, confIntE, confIntH0 = -1, -1, np.ones(len(hCompr))*(-1), -1, [-1,-1], [-1,-1]
        print(NORMAL + '')
        
    # return(E, H0, hPredict, R2, X2, confIntE, confIntH0, error)
    return(E, H0, hPredict, fPredict, comprMatSorted, R2, Chi2, residuals_h, residuals_f, error)




def get_Chi2(Ymeas, Ymodel, dof, err = 10):
    residuals = Ymeas-Ymodel
    # S = st.tstd(residuals)
    # S = (np.sum(residuals**2)/len(residuals))**0.5
    Chi2 = np.sum((residuals/err)**2)
    Chi2_dof = Chi2/dof
    return(Chi2_dof)


def testFun(cellId):
    
    #### MAIN PART [1/2]
    expDf = getExperimentalConditions(experimentalDataDir)
    tsDF = getCellTimeSeriesData(cellId)
    DIAMETER, hComprList, fComprList = segmentTimeSeries_meca(cellId, tsDF, expDf)
    Nc = len(hComprList)

    
    
    #### PLOT [1/2]
    nColsSubplot = 5
    nRowsSubplot = ((Nc-1) // nColsSubplot) + 1
    # First plot - fig & ax, gather all the F(h) curves, and will be completed later in the code
    fig, ax = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
    # Second plot - fig2 & ax2, 
    fig2, ax2 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
    # Third plot - fig3 & ax3, 
    fig3, ax3 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
    
    
    
    #### MAIN PART [2/2]
    for i in range(1,Nc+1): 
        hCompr, fCompr = hComprList[i-1], fComprList[i-1]
        E, H0, hPredict, fPredict, comprMatSorted, R2, Chi2, residuals_h, residuals_f, fitError = compressionFitChadwick_V2(hCompr, fCompr, DIAMETER)
        hComprSorted, fComprSorted = comprMatSorted[:, 0], comprMatSorted[:, 1]
        
        deltaCompr = (H0 - hCompr)/1000 # µm
        stressCompr = fCompr / (np.pi * (DIAMETER/2000) * deltaCompr)
        strainCompr = deltaCompr / (3*(H0/1000)) 
        strainPredict = stressCompr / E #((H0 - hPredict)/1000) / (3*(H0/1000))
        
        # Stats
        print(i, R2, Chi2)
    
        #### PLOT [2/2]
        colSp = (i-1) % nColsSubplot
        rowSp = (i-1) // nColsSubplot

        # ax2[i-1] with the 1 line plot
        if nRowsSubplot == 1:
            thisAx = ax[colSp]
        elif nRowsSubplot >= 1:
            thisAx = ax[rowSp,colSp]
            
        if nRowsSubplot == 1:
            thisAx2 = ax2[colSp]
        elif nRowsSubplot >= 1:
            thisAx2 = ax2[rowSp,colSp]
            
        if nRowsSubplot == 1:
            thisAx3 = ax3[colSp]
        elif nRowsSubplot >= 1:
            thisAx3 = ax3[rowSp,colSp]
            
        titleText = cellId + '__c' + str(i)
        
        thisAx.plot(hCompr,fCompr,'bo', ls='', markersize = 2)
        # thisAx.plot(hCompr,fCompr,'b-', linewidth = 0.8)
        # thisAx.plot(hComprSorted,fComprSorted,'r-', linewidth = 0.8)
        legendText = ''
        thisAx.set_xlabel('h (nm)')
        thisAx.set_ylabel('f (pN)')
        
        thisAx2.plot(stressCompr, strainPredict - strainCompr, 'b+')
        # thisAx2.legend(loc = 'upper right', prop={'size': 6})
        thisAx2.set_xlabel('h (pN)')
        # thisAx2.set_ylim([-50,+50])
        thisAx2.set_ylabel('residuals')
        
        thisAx3.plot(stressCompr,strainCompr,'kP', ls='', markersize = 4)
        legendText3 = ''
        thisAx3.set_xlabel('sigma (Pa)')
        thisAx3.set_ylabel('epsilon')
    
        if not fitError:
            legendText += 'H0 = {:.1f}nm\nE = {:.2e}Pa'.format(H0, E)
            # thisAx.plot(hComprSorted,fPredict,'y--', linewidth = 0.8, label = legendText)
            thisAx.plot(hPredict,fCompr,'k--', linewidth = 0.8, label = legendText)
            thisAx.legend(loc = 'upper right', prop={'size': 6})
            
            legendText3 += 'H0 = {:.1f}nm\nE = {:.2e}Pa'.format(H0, E)
            thisAx3.plot(stressCompr,strainPredict,'r+', linewidth = 0.8, label = legendText3)
            thisAx3.legend(loc = 'upper right', prop={'size': 6})

            
        else:
            titleText += '\nFIT ERROR'
    
        thisAx.title.set_text(titleText)
        
        axes = [thisAx, thisAx2, thisAx3]
        for axe in axes:
            axe.title.set_text(titleText)
            for item in ([axe.title, axe.xaxis.label, axe.yaxis.label] \
                         + axe.get_xticklabels() + axe.get_yticklabels()):
                item.set_fontsize(9)

    plt.show()

# %%% Main script


# cellId = '21-12-08_M2_P1_C1' # Best fits
# cellId = '21-12-08_M1_P2_C1' # Excellent fits
# cellId = '21-12-16_M1_P1_C10' # Good fits
# cellId = '21-12-16_M2_P1_C6' # Bad fits
# cellId = '21-01-21_M1_P1_C2' # Old fits with large h
cellId = '21-12-16_M1_P1_C3' # Bad fits
# cellId = '21-12-16_M1_P1_C2' # Bad fits
# cellId = '22-01-12_M1_P1_C2' # Recent fits
cellId = '21-01-18_M1-1_P1_C1'
testFun(cellId)


# %%% End

plt.close('all')






# %% compressionFitChadwick_StressStrain

# %%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm

import os
import time
import random
import traceback


from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
# from statannot import add_stat_annotation

pd.set_option('mode.chained_assignment',None)
pd.set_option('display.max_columns', None)

import re
dateFormatExcel = re.compile('\d{2}/\d{2}/\d{4}')
dateFormatOk = re.compile('\d{2}-\d{2}-\d{2}')

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'figure.autolayout': True})

# Add the folder to path
import sys
sys.path.append("C://Users//JosephVermeil//Desktop//ActinCortexAnalysis//Code_Python")
from getExperimentalConditions import getExperimentalConditions
from PincherAnalysis_JV import *

# 5. Global constants
SCALE_100X = 15.8 # pix/µm 
NORMAL  = '\033[0m'
RED  = '\033[31m' # red
GREEN = '\033[32m' # green
ORANGE  = '\033[33m' # orange
BLUE  = '\033[36m' # blue

# %%% Directories

mainDataDir = 'D://MagneticPincherData'
rawDataDir = os.path.join(mainDataDir, 'Raw')
depthoDir = os.path.join(rawDataDir, 'EtalonnageZ')
interDataDir = os.path.join(mainDataDir, 'Intermediate')
figureDir = os.path.join(mainDataDir, 'Figures')
timeSeriesDataDir = "C://Users//JosephVermeil//Desktop//ActinCortexAnalysis//Data_Analysis//TimeSeriesData"
experimentalDataDir = "C://Users//JosephVermeil//Desktop//ActinCortexAnalysis//Data_Experimental"

# %%% Data

expDf = getExperimentalConditions(experimentalDataDir)


# %%% Functions

def segmentTimeSeries_meca(f, tsDF, expDf):
    #### (0) Import experimental infos
    split_f = f.split('_')
    tsDF.dx, tsDF.dy, tsDF.dz, tsDF.D2, tsDF.D3 = tsDF.dx*1000, tsDF.dy*1000, tsDF.dz*1000, tsDF.D2*1000, tsDF.D3*1000
    thisManipID = split_f[0] + '_' + split_f[1]
    expDf['manipID'] = expDf['date'] + '_' + expDf['manip']
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]

    diameters = str(thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'])
    diameters = diameters.split('_')
    DIAMETER = int(diameters[0])
    EXPTYPE = str(thisExpDf.at[thisExpDf.index.values[0], 'experimentType'])

    Ncomp = max(tsDF['idxAnalysis'])
    
    loopStruct = thisExpDf.at[thisExpDf.index.values[0], 'loop structure'].split('_')
    nUplet = thisExpDf.at[thisExpDf.index.values[0], 'normal field multi images']
    if 'compression' in EXPTYPE:
        loop_totalSize = int(loopStruct[0])
        if len(loopStruct) >= 2:
            loop_rampSize = int(loopStruct[1])
        else:
            loop_rampSize = 0
        if len(loopStruct) >= 3:
            loop_excludedSize = int(loopStruct[2])
        else:
            loop_excludedSize = 0
        loop_ctSize = int((loop_totalSize - (loop_rampSize+loop_excludedSize))/nUplet)


    compField = thisExpDf.at[thisExpDf.index.values[0], 'ramp field'].split('_')
    minCompField = float(compField[0])
    maxCompField = float(compField[1])
    
    hComprList = []
    fComprList = []


    for i in range(1, Ncomp+1): #Ncomp+1):

        #### (1) Segment the compression n°i
        thisCompDf = tsDF.loc[tsDF['idxAnalysis'] == i,:]
        iStart = (findFirst(tsDF['idxAnalysis'], i))
        iStop = iStart+thisCompDf.shape[0]


        listB = thisCompDf.B.values
        
        # Test to check if most of the compression have not been deleted due to bad image quality 
        highBvalues = (listB > (maxCompField + minCompField)/2)
        N_highBvalues = np.sum(highBvalues)
        testHighVal = (N_highBvalues > 20)

        # Test to check if the range of B field is large enough
        minB, maxB = min(listB), max(listB)
        testRangeB = ((maxB-minB) > 0.7*(maxCompField - minCompField))
        thresholdB = (maxB-minB)/50
        thresholdDeltaB = (maxB-minB)/400

        # Is the curve ok to analyse ?
        doThisCompAnalysis = testHighVal and testRangeB # Some criteria can be added here

        if doThisCompAnalysis:
            #### (2) Inside the compression n°i, delimit the compression and relaxation phases

            # Delimit the start of the increase of B (typically the moment when the field decrease from 5 to 3)
            # and the end of its decrease (typically when it goes back from 3 to 5)
            
            try:
                # Correct for bugs in the B data
                if 'compressions' in EXPTYPE:
                    for k in range(1,len(listB)):
                        B = listB[k]
                        if B > 1.25*maxCompField:
                            listB[k] = listB[k-1]

                offsetStart, offsetStop = 0, 0
                minB, maxB = min(listB), max(listB)
                thresholdB = (maxB-minB)/50
                thresholdDeltaB = (maxB-minB)/400 # NEW CONDITION for the beginning of the compression : 
                # remove the first points where the steps in B are very very small

                k = 0
                while (listB[k] < minB+thresholdB) or (listB[k+1]-listB[k] < thresholdDeltaB):
                    offsetStart += int((listB[k] < minB+thresholdB) or ((listB[k+1]-listB[k]) < thresholdDeltaB))
                    k += 1

                k = 0
                while (listB[-1-k] < minB+thresholdB):
                    offsetStop += int(listB[-1-k] < minB+thresholdB)
                    k += 1

                jStart = offsetStart # Beginning of compression
                jMax = np.argmax(thisCompDf.B) # End of compression, beginning of relaxation
                jStop = thisCompDf.shape[0] - offsetStop # End of relaxation
            
            except:
                print(listB)
                print(testRangeB, thresholdB, thresholdDeltaB)

            # Four arrays
            hCompr = (thisCompDf.D3.values[jStart:jMax+1] - DIAMETER)
            fCompr = (thisCompDf.F.values[jStart:jMax+1])

            # Refinement of the compression delimitation.
            # Remove the 1-2 points at the begining where there is just the viscous relaxation of the cortex
            # because of the initial decrease of B and the cortex thickness increases.
            offsetStart2 = 0
            k = 0
            while (k<len(hCompr)-10) and (hCompr[k] < np.max(hCompr[k+1:min(k+10, len(hCompr))])):
                offsetStart2 += 1
                k += 1
            # Better compressions arrays
            hCompr = np.array(hCompr[offsetStart2:])
            fCompr = np.array(fCompr[offsetStart2:])

            hComprList.append(hCompr)
            fComprList.append(fCompr)
            
    return(DIAMETER, hComprList, fComprList)



def compressionFitChadwick(hCompr, fCompr, DIAMETER):
    
    error = False
    
    def chadwickModel(h, E, H0):
        R = DIAMETER/2
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0):
        R = DIAMETER/2
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)

    # some initial parameter values - must be within bounds
    initH0 = max(hCompr) # H0 ~ h_max
    initE = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
#     initH0, initE = initH0*(initH0>0), initE*(initE>0)
    
    initialParameters = [initE, initH0]
#     print(initialParameters)

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, 0)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]


    try:
        params, covM = curve_fit(inversedChadwickModel, fCompr, hCompr, initialParameters, bounds = parameterBounds)

        # Previously I fitted with y=F and x=H, but it didn't work so well cause H(t) isn't monotonous:
        # params, covM = curve_fit(chadwickModel, hCompr, fCompr, initialParameters, bounds = parameterBounds)
        # Fitting with the 'inverse Chadwick model', with y=H and x=F is more convenient

        E, H0 = params
        hPredict = inversedChadwickModel(fCompr, E, H0)
        err = dictSelectionCurve['Error']
        
        comprMat = np.array([hCompr, fCompr]).T
        comprMatSorted = comprMat[comprMat[:, 0].argsort()]
        hComprSorted, fComprSorted = comprMatSorted[:, 0], comprMatSorted[:, 1]
        fPredict = chadwickModel(hComprSorted, E, H0)
        
        # Stress and strain
        deltaCompr = (H0 - hCompr)/1000 # µm
        stressCompr = fCompr / (np.pi * (DIAMETER/2000) * deltaCompr)
        strainCompr = deltaCompr / (3*(H0/1000)) 
        strainPredict = stressCompr / (E*1e6) #((H0 - hPredict)/1000) / (3*(H0/1000))
        
        # residuals_h = hCompr-hPredict
        # residuals_f = fComprSorted-fPredict

        alpha = 0.975
        dof = len(fCompr)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = get_R2(hCompr, hPredict)
        
        Chi2 = get_Chi2(strainCompr, strainPredict, dof, err)

        varE = covM[0,0]
        seE = (varE)**0.5
        E, seE = E*1e6, seE*1e6
        confIntE = [E-q*seE, E+q*seE]
        confIntEWidth = 2*q*seE

        varH0 = covM[1,1]
        seH0 = (varH0)**0.5
        confIntH0 = [H0-q*seH0, H0+q*seH0]
        confIntH0Width = 2*q*seH0
        
        
    except:
        error = True
        E, H0, hPredict, R2, Chi2, confIntE, confIntH0 = -1, -1, np.ones(len(hCompr))*(-1), -1, -1, [-1,-1], [-1,-1]
    
    return(E, H0, hPredict, R2, Chi2, confIntE, confIntH0, error)







def compressionFitChadwick_StressStrain(hCompr, fCompr, H0, DIAMETER):
    
    error = False
    
    # def chadwickModel(h, E, H0):
    #     R = DIAMETER/2
    #     f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    #     return(f)

    # def inversedChadwickModel(f, E, H0):
    #     R = DIAMETER/2
    #     h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    #     return(h)
    
    def computeStress(f, h, H0):
        R = DIAMETER/2000
        delta = (H0 - h)/1000
        stress = f / (np.pi * R * delta)
        return(stress)
        
    def computeStrain(h, H0):
        delta = (H0 - h)/1000
        strain = delta / (3 * (H0/1000))
        return(strain)
    
    def constitutiveRelation(strain, K, stress0):
        stress = (K * strain) + stress0
        return(stress)
    
    def inversedConstitutiveRelation(stress, K, strain0):
        strain = (stress / K) + strain0
        return(strain)

    # some initial parameter values - must be within bounds
    initK = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    init0 = 0
    
    initialParameters = [initK, init0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, -np.Inf)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]


    try:
        strainCompr = computeStrain(hCompr, H0)
        stressCompr = computeStress(fCompr, hCompr, H0)
        
        params, covM = curve_fit(inversedConstitutiveRelation, stressCompr, strainCompr, initialParameters, bounds = parameterBounds)
        print(params)
        print(covM)
        # Previously I fitted with y=F and x=H, but it didn't work so well cause H(t) isn't monotonous:
        # params, covM = curve_fit(chadwickModel, hCompr, fCompr, initialParameters, bounds = parameterBounds)
        # Fitting with the 'inverse Chadwick model', with y=H and x=F is more convenient

        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stressCompr, K, strain0)
        err = dictSelectionCurve['Error']

        alpha = 0.975
        dof = len(stressCompr)-len(params)
        
        R2 = get_R2(strainCompr, strainPredict)
        
        Chi2 = get_Chi2(strainCompr, strainPredict, dof, err)

        varK = covM[0,0]
        seK = (varK)**0.5
        
        q = st.t.ppf(alpha, dof) # Student coefficient
        # K, seK = K*1e6, seK*1e6
        confIntK = [K-q*seK, K+q*seK]
        confIntKWidth = 2*q*seK
        
        
    except:
        error = True
        K, strainPredict, R2, Chi2, confIntK = -1, np.ones(len(strainCompr))*(-1), -1, -1, [-1,-1]
    
    return(stressCompr, strainCompr, K, strainPredict, R2, Chi2, confIntK, error)






def testFun(cellId):
    
    def chadwickModel(h, E, H0):
        R = DIAMETER/2
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0):
        R = DIAMETER/2
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)
    
    def computeStress(f, h, H0, D):
        R = D/2000
        delta = (H0 - h)/1000
        stress = f / (np.pi * R * delta)
        return(stress)
        
    def computeStrain(h, H0):
        delta = (H0 - h)/1000
        strain = delta / (3 * (H0/1000))
        return(strain)
    
    #### MAIN PART [1/2]
    expDf = getExperimentalConditions(experimentalDataDir)
    tsDF = getCellTimeSeriesData(cellId)
    DIAMETER, hComprList, fComprList = segmentTimeSeries_meca(cellId, tsDF, expDf)
    Nc = len(hComprList)    
    
    #### PLOT [1/2]
    nColsSubplot = 5
    nRowsSubplot = ((Nc-1) // nColsSubplot) + 1
    
    # 1st plot - fig & ax
    fig, ax = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
    
    # 2nd plot - fig & ax
    fig2, ax2 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
    
    
    #### MAIN PART [2/2]
    for i in range(1,Nc+1): 
        hCompr, fCompr = hComprList[i-1], fComprList[i-1]
        
        
        compressionStart_NbPts = 15
        hCompr_start = hCompr[:compressionStart_NbPts]
        fCompr_start = fCompr[:compressionStart_NbPts]
        
        E, H0, hPredict, R2, Chi2, confIntE, confIntH0, errorH = compressionFitChadwick(hCompr_start, fCompr_start, DIAMETER)
        
        stressCompr, strainCompr, K, strainPredict, R2, Chi2, confIntK, errorK = compressionFitChadwick_StressStrain(hCompr, fCompr, H0, DIAMETER)
        
        strainCompr = computeStrain(hCompr, H0)
        stressCompr = computeStress(fCompr, hCompr, H0, DIAMETER)
        
        #### PLOT [2/2]
        # 
        min_f = np.min(fCompr_start)
        low_f = np.linspace(0, min_f, 20)
        low_h = inversedChadwickModel(low_f, E/1e6, H0)
        
        # 
        colSp = (i-1) % nColsSubplot
        rowSp = (i-1) // nColsSubplot

        # 
        if nRowsSubplot == 1:
            thisAx = ax[colSp]
            thisAx2 = ax2[colSp]
        elif nRowsSubplot >= 1:
            thisAx = ax[rowSp,colSp]
            thisAx2 = ax2[rowSp,colSp]

            
        titleText = cellId + '__c' + str(i)
        
        
        thisAx2.plot(hCompr,fCompr,'bo', ls='', markersize = 2)
        legendText2 = ''
        thisAx2.set_xlabel('h (nm)')
        thisAx2.set_ylabel('f (pN)')
    
        if not errorH:
            legendText2 += 'H0 = {:.2f}nm'.format(H0)
            plot_startH = np.concatenate((low_h, hPredict))
            plot_startF = np.concatenate((low_f, fCompr_start))
            thisAx2.plot(plot_startH[0], plot_startF[0],'ro', markersize = 5)
            thisAx2.plot(plot_startH, plot_startF,'r--', linewidth = 0.8, label = legendText2)
            thisAx2.legend(loc = 'upper right', prop={'size': 6})
            
            
        thisAx.plot(stressCompr, strainCompr,'ko', ls='', markersize = 4)
        legendText = ''
        thisAx.set_xlabel('sigma (Pa)')
        thisAx.set_ylabel('epsilon')
            
        if not errorK:
            legendText += 'K = {:.2e}Pa'.format(K)
            thisAx.plot(stressCompr, strainPredict,'r--', linewidth = 0.8, label = legendText)
            thisAx.legend(loc = 'upper right', prop={'size': 6})

            
        else:
            titleText += '\nFIT ERROR'
    
        thisAx2.title.set_text(titleText)
        thisAx.title.set_text(titleText)
        
        axes = [thisAx, thisAx2]
        for axe in axes:
            axe.title.set_text(titleText)
            for item in ([axe.title, axe.xaxis.label, axe.yaxis.label] \
                         + axe.get_xticklabels() + axe.get_yticklabels()):
                item.set_fontsize(9)

    plt.show()
        
# %%% Main script


cellId = '22-02-09_M1_P1_C1'
testFun(cellId)


# %%% Check

cellId = '22-02-09_M1_P1_C1'
f = '22-02-09_M1_P1_C1_L40_20umdisc_PY.csv'
tsDF = getCellTimeSeriesData(cellId)

D, H, F = segmentTimeSeries_meca(f, tsDF, expDf)

# %%% End

plt.close('all')    


# %% Test plots

# %%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm

import os
import time
import random
import traceback


from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
# from statannot import add_stat_annotation

pd.set_option('mode.chained_assignment',None)
pd.set_option('display.max_columns', None)

import re
dateFormatExcel = re.compile('\d{2}/\d{2}/\d{4}')
dateFormatOk = re.compile('\d{2}-\d{2}-\d{2}')

import matplotlib
import matplotlib.pyplot as plt

# %%% Color and marker lists

colorList10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markerList10 = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']

bigPalette1 = sns.color_palette("tab20b")
bigPalette1_hex = bigPalette1.as_hex()

bigPalette2 = sns.color_palette("tab20c")
bigPalette2_hex = bigPalette2.as_hex()

colorList30 = []
for ii in range(2, -1, -1):
    colorList30.append(bigPalette2_hex[4*0 + ii]) # blue
    colorList30.append(bigPalette2_hex[4*1 + ii]) # orange
    colorList30.append(bigPalette2_hex[4*2 + ii]) # green
    colorList30.append(bigPalette1_hex[4*3 + ii]) # red
    colorList30.append(bigPalette2_hex[4*3 + ii]) # purple
    colorList30.append(bigPalette1_hex[4*2 + ii]) # yellow-brown
    colorList30.append(bigPalette1_hex[4*4 + ii]) # pink
    colorList30.append(bigPalette1_hex[4*0 + ii]) # navy    
    colorList30.append(bigPalette1_hex[4*1 + ii]) # yellow-green
    colorList30.append(bigPalette2_hex[4*4 + ii]) # gray

# %%% Imports

N = 3
fig, ax = plt.subplots(N, N, figsize = (3*N, 3*N))
for ii in range(N):
    for jj in range(N):
        color = colorList30[3*ii + jj]
        marker = markerList10[3*ii + jj]
        ax[ii,jj].plot([1,2,3,4], [1,2,3,4], color = color, marker = marker, mec = 'k', ms = 8, ls = '', label = str(3*ii + jj))
        ax[ii,jj].set_xlim([0,5])
        ax[ii,jj].set_ylim([0,5])
fig.legend(loc='upper center', bbox_to_anchor=(0.5,0.95), ncol = N*N)
fig.suptitle('test')
# fig.tight_layout()
fig.show()

# %% Color test !

NORMAL  = '\033[1;0m'
RED  = '\033[0;31m' # red
GREEN = '\033[1;32m' # green
ORANGE  = '\033[0;33m' # orange
BLUE  = '\033[0;36m' # blue
YELLOW = '\033[1;33m'

print(NORMAL + 'yellow' + NORMAL)
print(RED + 'yellow' + NORMAL)
print(GREEN + 'yellow' + NORMAL)
print(ORANGE + 'yellow' + NORMAL)
print(YELLOW + 'yellow' + NORMAL)
print(BLUE + 'yellow' + NORMAL)

print("\n")

# print("\033[0;37;48m Normal text\n")
# print("\033[2;37;48m Underlined text\033[0;37;48m \n")
# print("\033[1;37;48m Bright Colour\033[0;37;48m \n")
# print("\033[3;37;48m Negative Colour\033[0;37;48m \n")
# print("\033[5;37;48m Negative Colour\033[0;37;48m\n")
# print("\033[1;37;40m \033[2;37:40m TextColour BlackBackground          TextColour GreyBackground                WhiteText ColouredBackground\033[0;37;40m\n")
# print("\033[1;30;40m Dark Gray      \033[0m 1;30;40m            \033[0;30;47m Black      \033[0m 0;30;47m               \033[0;37;41m Black      \033[0m 0;37;41m")
# print("\033[1;31;40m Bright Red     \033[0m 1;31;40m            \033[0;31;47m Red        \033[0m 0;31;47m               \033[0;37;42m Black      \033[0m 0;37;42m")
# print("\033[1;32;40m Bright Green   \033[0m 1;32;40m            \033[0;32;47m Green      \033[0m 0;32;47m               \033[0;37;43m Black      \033[0m 0;37;43m")
# print("\033[1;33;48m Yellow         \033[0m 1;33;48m            \033[0;33;47m Brown      \033[0m 0;33;47m               \033[0;37;44m Black      \033[0m 0;37;44m")
# print("\033[1;34;40m Bright Blue    \033[0m 1;34;40m            \033[0;34;47m Blue       \033[0m 0;34;47m               \033[0;37;45m Black      \033[0m 0;37;45m")
# print("\033[1;35;40m Bright Magenta \033[0m 1;35;40m            \033[0;35;47m Magenta    \033[0m 0;35;47m               \033[0;37;46m Black      \033[0m 0;37;46m")
# print("\033[1;36;40m Bright Cyan    \033[0m 1;36;40m            \033[0;36;47m Cyan       \033[0m 0;36;47m               \033[0;37;47m Black      \033[0m 0;37;47m")
# print("\033[1;37;40m White          \033[0m 1;37;40m            \033[0;37;40m Light Grey \033[0m 0;37;40m               \033[0;37;48m Black      \033[0m 0;37;48m")

# %% Next test !



