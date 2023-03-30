# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 18:48:52 2023

@author: JosephVermeil
"""

#%% Imports

import os
import cv2
import shutil
import traceback

import numpy as np
# import pyjokes as pj
import matplotlib.pyplot as plt

from skimage import io


import numpy as np
import scipy.interpolate as interp
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import data, img_as_float, util, exposure, filters, segmentation

#### Local Imports

import sys
import CortexPaths as cp
os.chdir(cp.DirRepoPython)

import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun


#%% Define directories

mainDir = "C://Users//JosephVermeil//Desktop//Alice_Movies"
cellDir = 'Cells'

#%% Import image

C1File = 'C1_8um_02_movie_bleach-corr.tif'
C1_path = os.path.join(mainDir, cellDir, C1File)

C2File = 'C2_8um_02_movie_bleach-corr.tif'
C2_path = os.path.join(mainDir, cellDir, C2File)

C3File = 'C3_8um_02_movie_bleach-corr.tif'
C3_path = os.path.join(mainDir, cellDir, C3File)

C4File = 'C4_8um_02_movie_bleach-corr.tif'
C4_path = os.path.join(mainDir, cellDir, C4File)

Iraw = io.imread(C1_path)
nz, ny, nx = Iraw.shape

#%% Import image 20um

C1File = 'C1_20um_07_movie_bleach-corr-3.tif'
C1_path = os.path.join(mainDir, cellDir, C1File)

C2File = 'C2_20um_07_movie_bleach-corr-3.tif'
C2_path = os.path.join(mainDir, cellDir, C2File)

C3File = 'C3_20um_07_movie_bleach-corr-3.tif'
C3_path = os.path.join(mainDir, cellDir, C3File)

Iraw = io.imread(C3_path)
nz, ny, nx = Iraw.shape

#%% Open and draw initial snake




# %%




# %%

def store_evolution_in(lst):
    """
    Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


# Morphological ACWE
# image = img_as_float(data.camera())

# # Initial level set
# init_ls = checkerboard_level_set(image.shape, 6)
# # List with intermediate results for plotting the evolution
# evolution = []
# callback = store_evolution_in(evolution)
# ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
#                              smoothing=3, iter_callback=callback)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

# ax[0].imshow(image, cmap="gray")
# ax[0].set_axis_off()
# ax[0].contour(ls, [0.5], colors='r')
# ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

# ax[1].imshow(ls, cmap="gray")
# ax[1].set_axis_off()
# contour = ax[1].contour(evolution[2], [0.5], colors='g')
# contour.collections[0].set_label("Iteration 2")
# contour = ax[1].contour(evolution[7], [0.5], colors='y')
# contour.collections[0].set_label("Iteration 7")
# contour = ax[1].contour(evolution[-1], [0.5], colors='r')
# contour.collections[0].set_label("Iteration 35")
# ax[1].legend(loc="upper right")
# title = "Morphological ACWE evolution"
# ax[1].set_title(title, fontsize=12)


# Morphological GAC
# image = img_as_float(data.coins())
# gimage = inverse_gaussian_gradient(image)
image = img_as_float(Iraw[0])
# gimage = (inverse_gaussian_gradient(image)) # util.invert
gimage = util.invert(image) # util.invert
vmin, vmax = np.min(gimage), np.max(gimage)
gimage_rescale = exposure.rescale_intensity(gimage, in_range=(vmin, vmax))

ny, nx = gimage_rescale.shape
Y0, X0 = np.arange(0, ny, dtype=int), np.arange(0, nx, dtype=int)
ff = interp.RegularGridInterpolator((X0, Y0), gimage_rescale.T, method='linear', bounds_error=True)

Y1, X1 = np.arange(ny-1, step=0.2), np.arange(nx-1, step=0.2)
XX1, YY1 = np.meshgrid(X1, Y1, indexing='ij')
gimage_interp = ff((XX1, YY1))
gimage_interp = gimage_interp.T

th = filters.threshold_otsu(gimage_interp)
image2 = util.invert(gimage_interp)

ax[0].imshow(image2, cmap="gray")
# ax[1].imshow(gimage_rescale, cmap="gray")
ax[1].imshow(gimage_interp, cmap="gray")



# Initial level set
init_ls = np.zeros(gimage_interp.shape, dtype=np.int8)
init_ls[460:560, 450:550] = 1
# init_ls[10:-10, 10:-10] = 1
# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls_0 = segmentation.morphological_geodesic_active_contour(gimage_interp, num_iter=500,
                                            init_level_set=init_ls,
                                            smoothing=1, balloon=+1,
                                            threshold=th,
                                            iter_callback=callback)

ls_01 = ndi.binary_erosion(ls_0, iterations=5)
ls_02 = ndi.binary_dilation(ls_01, iterations=5)

ax[2].imshow(image2, cmap="gray")
ax[2].set_axis_off()
ax[2].contour(ls_0, [0.5], colors='r')
ax[2].set_title("Morphological GAC segmentation", fontsize=12)

ax[3].imshow(ls_0, cmap="gray")
ax[3].set_axis_off()
contour = ax[3].contour(evolution[0], [0.5], colors='g')
contour.collections[0].set_label("Iteration 0")
contour = ax[3].contour(evolution[20], [0.5], colors='y')
contour.collections[0].set_label("Iteration 20")
contour = ax[3].contour(ls_02, [0.5], colors='r')
contour.collections[0].set_label("Iteration 50")
ax[3].legend(loc="upper right")
title = "Morphological GAC evolution"
ax[3].set_title(title, fontsize=12)

fig.tight_layout()
plt.show()

# %%

previous_ls = ls_02

fig, axes = plt.subplots(4, 5, figsize=(18, 15))
ax = axes.flatten()
listContours = []

for k in range(1,80):
    image = img_as_float(Iraw[k])
    gimage = util.invert(image)
    vmin, vmax = np.min(gimage), np.max(gimage)
    gimage_rescale = exposure.rescale_intensity(gimage, in_range=(vmin, vmax))

    th = filters.threshold_otsu(gimage_rescale)
    
    # Initial level set
    init_ls = ndi.binary_erosion(previous_ls, iterations=5)

    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = segmentation.morphological_geodesic_active_contour(gimage_rescale, num_iter=15,
                                                init_level_set=init_ls,
                                                smoothing=0, balloon=+1,
                                                threshold=th,
                                                iter_callback=callback)
    
    ls_1 = ndi.binary_erosion(ls, iterations=3)
    ls_2 = ndi.binary_dilation(ls_1, iterations=2)
    
    previous_ls = ls_2
    
    if (k-1)%4 == 0:
        kk = (k-1)//4
        ax[kk].imshow(image, cmap="gray")
        ax[kk].set_axis_off()
        # ax[kk].imshow(gimage_rescale, cmap="gray")
        ax[kk].contour(init_ls, [0.5], colors='g')
        # contour.collections[0].set_label("Iteration 0")
        ax[kk].contour(evolution[3], [0.5], colors='y')
        # contour.collections[0].set_label("Iteration 10")
        ax[kk].contour(ls_2, [0.5], colors='r')
        # contour.collections[0].set_label("Iteration 20")
        # ax[kk].legend(loc="upper right")

fig.tight_layout()
plt.show()

# %%

y, x = np.arange(0, ny, dtype=int), np.arange(0, nx, dtype=int)
def ff(x, y):
    return x**2 + y**2
xg, yg = np.meshgrid(x, y, indexing='ij')

data = ff(xg, yg)
fi = interp.RegularGridInterpolator((x, y), data,
                                 bounds_error=False, fill_value=None)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xg.ravel(), yg.ravel(), data.ravel(),
           s=60, c='k', label='data')


xx = np.linspace(-4, 9, 31)
yy = np.linspace(-4, 9, 31)
X, Y = np.meshgrid(xx, yy, indexing='ij')
# interpolator
ax.plot_wireframe(X, Y, fi((X, Y)), rstride=3, cstride=3,
                  alpha=0.4, color='m', label='linear interp')
# ground truth
ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
                  alpha=0.4, label='ground truth')
plt.legend()
plt.show()