# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:57:30 2024

@author: Elo_Pro
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
#%%
"""
use the data from 16/05 to train the neuron network
2D data x=EGFP, y=DsRed, ground truth : Color
Green = 0
Yellow = 1
Red = 2
"""
path="C:/Users/BioMecaCell/Desktop/CortExplore/Code_Python/Code_EE/DL"
path_train=os.path.join(path,'data_fluo_24.05.23.txt')
path_test=os.path.join(path,'data_fluo_24.06.04_wobg.txt')

#%%
path04='D:/Eloise/MagneticPincherData/Raw/'
file04=os.path.join(path04,'data_24.06.04_new.txt')
#data04=pd.read_csv(file04, sep='\t')
#data04=data04.rename({'Unnamed: 0':'i'},axis='columns')

path12='D:/Eloise/MagneticPincherData/Raw/24.06.12'
file12=os.path.join(path12,'data_24.06.12.txt')
data12=pd.read_csv(path12 + '/data_24.06.12.txt', sep='\t')
data12=data12.rename({'Unnamed: 0':'i'},axis='columns')

path05='D:/Eloise/MagneticPincherData/Raw/24.05.23'
file05=os.path.join(path05,'data_24.05.23vf.txt')
data05=pd.read_csv(path05 + '/data_24.05.23vf.txt', sep='\t')
data05=data05.rename({'Unnamed: 0':'i'},axis='columns')
data05= data05[data05['manipID'].str.contains('24-05-23_M1') ==True]
#%%
def load_data(filename):
  with open(filename) as file :
    reader = pd.read_csv(file, sep='\t')
    reader.loc[reader['Color']=='Green',['gt']]=0
    reader.loc[reader['Color']=='Yellow',['gt']]=1
    reader.loc[reader['Color']=='Red',['gt']]=2
    data=pd.DataFrame.to_numpy(reader[['EGFP','DsRed','gt']])
    data = np.asarray(data).astype(float)
    x,y_gt=[],[]
    for row in range(len(data)):
        x0=np.array(data[row,:-1],dtype=float)
        x.append(x0.reshape(2,1))
        y_gt.append(int(data[row,-1]))
  num_samples=data.shape[0]
  y=np.reshape(data[:,-1],(num_samples,1))
  xt=data[:,:-1]
  return data,xt,y_gt
#%%
data04,x04,y_gt04=load_data(file04)


data,x,y_gt=load_data(path_test)

#%%
def plot_data_2d(x,y_gt):
  plt.scatter(x[:,0],x[:,1], c=y_gt, s=50)
  plt.colorbar()
  plt.axis("equal")
  plt.xlabel('EGFP')
  plt.ylabel('DsRed')
  plt.xscale('log')
  plt.yscale('log')
  #plt.xlim(100,10**4)
  #plt.ylim(100,10**4)
  plt.show()
  
#%%
plot_data_2d(x04, y_gt04)

#%%
def normalize_data(x,y):
    gmax=np.max([k[0]for k in x])
    rmax=np.max([l[1]for l in x])
    for i in range(len(x)):
        x[i][0]=x[i][0]/gmax
        x[i][1]=x[i][1]/rmax
    return x,y/np.max(y)
    
#%%
x04,y_gt04=normalize_data(x04,y_gt04)
#%%
def sigmoid(x):
  return 1/(1+np.exp(-x))

def dnn2_class(wa, wb, a, b, x):
  #1st layer ?
  x_a = x
  p_a = x_a @ wa + a
  y_a = sigmoid(p_a)

  #2nd layer ?
  x_b = y_a
  p_b = x_b @ wb + b
  y_b = sigmoid(p_b)

  
  y_p=y_b

  return y_p
#%%
def dnn3_class(wa, wb,wc, a, b,c, x):
  #1st layer ?
  x_a = x
  p_a = x_a @ wa #+ a
  y_a = sigmoid(p_a)

  #2nd layer ?
  x_b = y_a
  p_b = x_b @ wb #+ b
  y_b = sigmoid(p_b)
  
  #3rd layer ?
  x_c = y_b
  p_c = x_c @ wc #+ c
  y_c = sigmoid(p_c)
  
  y_p=y_c

  return y_p
#%%
from numpy.random import default_rng
rng=default_rng()

num_neurons = 3
wa = rng.standard_normal(size=(2,num_neurons))
wb = rng.standard_normal(size=(num_neurons,3))
wc = rng.standard_normal(size=(3,1))
a = rng.standard_normal()
b = rng.standard_normal()
c = rng.standard_normal()

#%%
def plot_pred_2d(x,y_gt,y_p):
  plt.scatter(x[:,0],x[:,1], c=np.abs(y_gt-1), s=50,label='ground truth',cmap='RdYlGn')
  plt.scatter(x[:,0],x[:,1], c=[np.abs(i[0]-1) for i in y_p], s=100, marker='x', label='predictions',cmap='RdYlGn')
  plt.colorbar()
  plt.legend()
  plt.axis("equal")
  plt.xlabel('EGFP')
  plt.ylabel('DsRed')
  plt.xscale('log')
  plt.yscale('log')
  plt.title('True and predicted colors for cells from 16/05/24 - 3 layers neuronal network')
  plt.show()

#%%

plot_pred_2d(x04, y_gt04, y_p=dnn3_class(wa, wb,wc,a,b,c, x04))

#%%
def d_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))
#%%
num_samples=len(x)
num_train_interactions=10 **5
#needs a lot more iterations bc of the number of weights
#here 9 trainable parameters (2x3 + 3)
eta=.1 # learning rate

for i in range(num_train_interactions):
  #need to pick sample randomly
  selected=rng.integers(0,num_samples)
  x_selected=np.reshape(x04[selected],(1,-1))
  #to multiply 2D arrays we need 2D arrays
  y_gt_selected=np.reshape(y_gt04[selected],(1,-1))

  x_selected_a = x_selected
  p_a = x_selected_a @ wa #+ a
  y_selected_a = sigmoid(p_a)

  x_selected_b = y_selected_a
  p_b = x_selected_b @ wb #+ b
  y_selected_b = sigmoid(p_b)
  
  x_selected_c = y_selected_b
  p_c = x_selected_c @ wc #+ c
  y_selected_c = sigmoid(p_c)

  y_p_selected = y_selected_c

  #error calculation
  error=y_p_selected - y_gt_selected

  #backward pass
  
  delta_c = error * d_sigmoid(p_c)
  #sum to collect the info from the layer
  wc = wc - eta * delta_c * np.transpose(x_selected_c)
  #c = c - eta * error
  
  delta_b = np.sum(wc * delta_c, axis=1) * d_sigmoid(p_b)
  #sum to collect the info from the layer
  wb = wb - eta * delta_b * np.transpose(x_selected_b)
  #b = b - eta * error
  
  delta_a = np.sum(wb * delta_b, axis=1) * d_sigmoid(p_a)
  #sum to collect the info from the layerb
  wa = wa - eta * delta_a * np.transpose(x_selected_a)
  #a = a - eta * error
  

  
  if i % 100 == 0:
    print(f'i={i} y_p={y_p_selected} error={error}')
#%%
plot_pred_2d(x04,y_gt04,y_p=dnn3_class(wa,wb,wc,a,b,c,x04))
#%%
x1,y1=[],[]
for row in range(len(data)):
    x0=np.array(data[row,:-1],dtype=np.float32)
    x1.append(x0.reshape(2,1))
    y1.append(int(data[row,-1]))

#%%
mlp_template = dl.MultiLayerPerceptron(in_features=2*1,hidden_features=[10,10], out_features=3)
mlp_template[..., "activation"].configure(Sigmoid)
mlp_model = mlp_template.create()

print(mlp_model)

print(f"{sum(p.numel() for p in mlp_model.parameters())} trainable parameters")

# %%
classifier_template = dl.Classifier(model=mlp_template, num_classes=3,
                                    make_targets_one_hot=True, loss=MSELoss(), optimizer=dl.SGD(lr=1))
classifier = classifier_template.create()

print(classifier)

#%%
train=list(zip(x1,y1))
train_dataloader=dl.DataLoader(train)

#%%
trainer= dl.Trainer(max_epochs=1, accelerator="auto")

#%%
trainer.fit(classifier, train_dataloader)

#%%
trainer.test(classifier,train_dataloader)

#%%
def plot_confusion_matrix(classifier,trainer,dataloader):
    """plot confusion matrix"""
    confusion_matrix=np.zeros((3,3),dtype=int)
    for cell, gt_color in dataloader:
        pred=classifier(cell)
        max_pred, pred_color=pred.max(dim=1)
        np.add.at(confusion_matrix,(gt_color,pred_color),1)
        
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix,annot=True,square=True,cmap=sns.cubehelix_palette(light=0.95,as_cmap=True),vmax=100)
    plt.xlabel("pred color")
    plt.ylabel("ground truth color")
    plt.show()
    return confusion_matrix
    
#%%
confu=plot_confusion_matrix(classifier, trainer, train_dataloader)


