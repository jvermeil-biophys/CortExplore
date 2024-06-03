# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:26:05 2024

@author: JosephVermeil
"""

# %% Imports

import os
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import skimage as skm
import scipy.ndimage as ndi
import pandas as pd

import UtilityFunctions as ufun

# %% Try on an example

src_dir = "D://MagneticPincherData//Raw//24.02.26_NanoIndent_ChiaroData_25um//M2_Films//Reslices"
src_file = "C1-1_4.tiff"
src_path = os.path.join(src_dir, src_file)

Kraw = skm.io.imread(src_path)
[ny, nx] = Kraw.shape
xx = np.arange(nx)

profile_raw = Kraw[ny//2, :]

fig1, axes1 = plt.subplots(2, 1, figsize = (8, 8))
ax = axes1[0]
ax.imshow(Kraw, cmap='Greys_r')
ax.axhline(ny//2, c='r', ls='--', lw=1)

ax = axes1[1]
ax.plot(xx, profile_raw)


Kblur = skm.filters.gaussian(Kraw, sigma=(0, 2))
profile_blur = Kblur[ny//2, :]

fig2, axes2 = plt.subplots(2, 1, figsize = (8, 8))
ax = axes2[0]
ax.imshow(Kblur, cmap='Greys_r')
ax.axhline(ny//2, c='r', ls='--', lw=1)

ax = axes2[1]
ax.plot(xx, profile_blur)

left, right = profile_blur[:nx//2], profile_blur[nx//2:]
i_left, i_right = np.argmax(left), nx//2 + np.argmax(right)

Kedges = skm.filters.sobel_v(Kblur)
profile_edges = Kedges[ny//2, :]

fig3, axes3 = plt.subplots(2, 1, figsize = (8, 8))
ax = axes3[0]
ax.imshow(Kedges, cmap='Greys_r')
ax.axhline(ny//2, c='r', ls='--', lw=1)

ax = axes3[1]
ax.plot(xx, profile_edges)

left, right = profile_edges[:nx//2], profile_edges[nx//2:]
i_left2, i_right2 = np.argmax(left), nx//2 + np.argmax(-right)

ax = axes1[1]
ax.axvline(i_left, c='g', ls='--', lw=1)
ax.axvline(i_right, c='g', ls='--', lw=1)
ax = axes2[1]
ax.axvline(i_left, c='g', ls='--', lw=1)
ax.axvline(i_right, c='g', ls='--', lw=1)
ax = axes3[1]
ax.axvline(i_left, c='g', ls='--', lw=1)
ax.axvline(i_right, c='g', ls='--', lw=1)

ax = axes1[1]
ax.axvline(i_left2, c='b', ls='--', lw=1)
ax.axvline(i_right2, c='b', ls='--', lw=1)
ax = axes2[1]
ax.axvline(i_left2, c='b', ls='--', lw=1)
ax.axvline(i_right2, c='b', ls='--', lw=1)
ax = axes3[1]
ax.axvline(i_left2, c='b', ls='--', lw=1)
ax.axvline(i_right2, c='b', ls='--', lw=1)


plt.show()


# %% Try on an example

src_dir = "D://MagneticPincherData//Raw//24.02.26_NanoIndent_ChiaroData_25um//M2_Films//Reslices"
src_file = "C1_1.tif"
src_path = os.path.join(src_dir, src_file)

Kraw = skm.io.imread(src_path)[:100]
Kblur = skm.filters.gaussian(Kraw, sigma=(0, 4))
Kedges = skm.filters.sobel_v(Kblur)
[ny, nx] = Kraw.shape
xx = np.arange(nx)
yy = np.arange(ny)

i_left, i_right = np.zeros(ny), np.zeros(ny)

for y in range(ny):
    profile_blur = Kblur[y, :]
    profile_blur = profile_blur/np.max(profile_blur)
    # left, right = profile_blur[:nx//2], profile_blur[nx//2:]
    # i_left[y], i_right[y] = np.argmax(left), nx//2 + np.argmax(right)
    
    profile_edges = Kedges[y, :]
    profile_edges = profile_edges/np.max(profile_edges)
    # left, right = profile_edges[:nx//2], profile_edges[nx//2:] # nx//2 # 3*nx//4 # 2*nx//3
    # i_left[y], i_right[y] = np.argmax(left), nx//2 + np.argmax(-right)

    a = 0.2
    left  = profile_edges[:nx//2] + a*profile_blur[:nx//2]
    right = profile_edges[390:] - a*profile_blur[390:] # nx//2 # 3*nx//4 # 2*nx//3
    i_left[y], i_right[y] = np.argmax(left), 390 + np.argmax(-right)
    
    
fig1, axes1 = plt.subplots(2, 1, figsize = (8, 8))
ax = axes1[0]
ax.imshow(Kblur, cmap='Greys_r')

ax = axes1[1]
ax.imshow(Kblur, cmap='Greys_r')
# ax.plot(i_left, yy, c='r', ls='-', lw=1)
# ax.plot(i_right, yy, c='r', ls='-', lw=1)
ax.plot(i_left, yy, c='b', ls='-', lw=1)
ax.plot(i_right, yy, c='b', ls='-', lw=1)

plt.tight_layout()
plt.show()


# %% Make into a table

SCALE_TIME = 2 # Fps
SCALE_SPACE = 15.8 # pix/um

yy = yy
x1 = i_left
x2 = i_right
T = yy/SCALE_TIME
R = 0.5*np.abs(x2-x1)/SCALE_SPACE

df = pd.DataFrame({'y':yy, 'x1':x1, 'x2':x2,
                   'T':T, 'R':R})

# df.plot('T', 'R')

R_init = np.median(R[:25])
# R_init = np.min(R[:ny//2])
# R_init = np.percentile(R[:20], 1)
R_high = np.percentile(R, 85)

print(R_init, R_high)

slope_if = ny//3 + ufun.findFirst(True, R[ny//3:] >= 0.995*R_high) + 0
slope_ii = slope_if - ufun.findFirst(True, R[:slope_if][::-1] <= 1.000*R_init) - 1
plateau_if = len(R) - ufun.findFirst(True, R[::-1] >= R_high)

ii = np.arange(len(R))
zone = (ii > slope_ii).astype(int) + (ii > slope_if).astype(int) + (ii > plateau_if).astype(int)
color_list = plt.cm.rainbow(np.linspace(0.2, 0.8, 4))[::-1]
color_col = np.array([matplotlib.colors.rgb2hex(color_list[zone[k]]) for k in range(len(R))])

df['zone'] = zone

params, results = ufun.fitLineHuber(T[zone == 1], R[zone == 1])
print(params)
R_fit = T[zone == 1]*params[1] + params[0]

R_init2 = np.mean(R[zone == 0])
# R_init2 = R_init
R_high2 = np.mean(R[zone == 2])
T_init = (R_init2-params[0])/params[1]
T_high = (R_high2-params[0])/params[1]

fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.scatter(T, R, c=color_col, 
        marker = 'o', edgecolor='k', s = 50)
ax.plot(T[zone == 1], R_fit, 'b--', label = f'R = {params[1]:.2f}*T + {params[0]:.1f}')
ax.axhline(R_init2, c='r', ls='--', lw=1)
ax.axhline(R_high2, c='r', ls='--', lw=1)
ax.plot([T_init, T_high], [R_init2, R_high2], 'ro')
# ax.set_ylim([0, 1.2*ax.get_ylim()[1]])
ax.set_xlabel('Time (s)')
ax.set_ylabel('R0 (µm)')

ax.legend()

plt.show()

## %% Save the table

# date, manip, cell, indent = '24.02.26', 'M3', src_file.split('_')[0], '002'

# dst_dir = f'D://MagneticPincherData//Raw//{date}_NanoIndent_ChiaroData_25um//{manip}//{cell}//FilmData' #'Indentations//{cell}-20um Indentation_{indent}.txt'
# if not os.path.exists(dst_dir):
#     os.makedirs(dst_dir)

# dst_name = 'df_' + src_file.split('.')[0] + f'_I{int(indent):.0f}' + '.csv'

# dst_path = os.path.join(dst_dir, dst_name)

# df.to_csv(dst_path)



