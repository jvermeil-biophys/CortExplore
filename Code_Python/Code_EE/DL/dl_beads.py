# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:24:22 2024

@author: ee
"""

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import deeplay as dl
import torch.nn as nn
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import MSELoss
import pandas as pd
import seaborn as sns
import deeptrack as dt
import torch
#%%
"""
    simulate beads data with depthograph to train network
"""

bead=dt.Sphere(position=np.array([0.5,0.5])*64,position_unit="pixel",radius=500*dt.units.nm,
               refractive_index=1.45+0.02j)

bf_microscope=dt.Brightfield(wavelength=500*dt.units.nm, NA=1.0,
                             resolution=1*dt.units.um, magnification=10,
                             refractive_index_medium=1.33, upsample=2, output_region=(0,0,64,64))

#%%
illuminated_sample=bf_microscope(bead)

clean_bead=illuminated_sample >> dt.NormalizeMinMax() \
    >> dt.MoveAxis(2,0) >> dt.pytorch.ToTensor(dtype=torch.float)

#%%
noise=dt.Poisson(snr=lambda: 2.0 +np.random.rand())
noisy_bead=illuminated_sample >> noise >> dt.NormalizeMinMax() \
    >> dt.MoveAxis(2,0) >> dt.pytorch.ToTensor(dtype=torch.float)

#%%
def plot_image(title,image):
    plt.imshow(image,cmap="gray")
    plt.title(title)
    plt.axis('off')
    plt.show()
#%%
pip= noisy_bead & clean_bead
for i in range(2):
    plt.figure()
    inpit,target=pip.update().resolve()
    plot_image(f"input image {i}",inpit.permute(1,2,0))
    plt.figure()
    plot_image(f"target {i}",target.permute(1,2,0))
    
#%%
class SimulatedBeadsDataset(torch.utils.data.Dataset):
    """Simulate dataset for the beads using depthograph ?"""
    
    def __init__(self,pip,buffer_size,replace=0):
        """initialize dataset"""
        self.pip,self.buffer_size,self.replace=pip,buffer_size,replace
        self.images=[pip.update().resolve() for _ in range(buffer_size)]
        
    def __len__(self):
        """return size of dataset buffer"""
        return self.buffer_size
    
    def __getitem__(self,idx):
        """retrieve a noisy-clean image pair from dataset"""
        if np.random.rand()<self.replace:
            self.images[idx]=self.pip.update().resolve()
        image_pair=self.images[idx]
        noisy_image,clean_image=image_pair[0],image_pair[1]
        return noisy_image,clean_image
    
#%%
dataset=SimulatedBeadsDataset(pip, buffer_size=256,replace=0.1)
loader=torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True)
    










