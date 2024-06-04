# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:58:48 2024

@author: JosephVermeil
"""

# %% 1. Import

import tensorflow as tf
from tensorflow.keras.datasets import mnist

import autokeras as ak

import skimage as skm
import matplotlib.pyplot as plt


# %% 2. Do

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:100]
y_train = y_train[:100]
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# %% 3. Do

viz_x = x_train[0].astype(int)

fig, ax = plt.subplots(1,1)
ax.imshow(viz_x)
plt.show()