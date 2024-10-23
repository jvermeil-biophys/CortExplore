# -*- coding: utf-8 -*-
"""DL_MNIST.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kZ8jQT8E1yBrqrJcmCcktAUy8zGRBEx-
"""
#%%
import os

if not os.path.exists("MNIST_dataset"):
    os.system("git clone https://github.com/DeepTrackAI/MNIST_dataset")
    
#%%
path_origin='C:/Users/BioMecaCell/Desktop/CortExplore/Code_Python/Code_EE/DL'
#%%
train_path = os.path.join(path_origin,"MNIST_dataset", "mnist", "train")
#%%
train_images_files = sorted(os.listdir(train_path))

#%%
import matplotlib.pyplot as plt
train_images = []
for file in train_images_files:
  image = plt.imread(os.path.join(train_path, file))
  train_images.append(image)

print(train_images[0].shape)

#%%
train_digits = []
for file in train_images_files:
  filename = os.path.basename(file)
  digit = int(filename[0])
  train_digits.append(digit)

print(len(train_digits))
#%%
import numpy as np

figs,axs=plt.subplots(nrows=3, ncols=10, figsize=(20,6))
for ax in axs.ravel():
  idx_image=np.random.choice(len(train_digits))
  ax.imshow(train_images[idx_image],cmap="Greys")
  ax.set_title(f"Label: {train_digits[idx_image]}", fontsize=20)
  ax.axis("off")
plt.show()

#%%

import deeplay as dl
from torch.nn import Sigmoid
mlp_template = dl.MultiLayerPerceptron(in_features=28 * 28, hidden_features=[32, 32], out_features=10)
mlp_template[..., "activation"].configure(Sigmoid)
mlp_model = mlp_template.create()

print(mlp_model)

print(f"{sum(p.numel() for p in mlp_model.parameters())} trainable parameters")


#%%
from torch.nn import MSELoss

classifier_template = dl.Classifier(model=mlp_template, num_classes=10,
                                    make_targets_one_hot=True, loss=MSELoss(), optimizer=dl.SGD(lr=.1))
classifier = classifier_template.create()

print(classifier)
#%%
train_images_digits=list(zip(train_images, train_digits))
train_dataloader=dl.DataLoader(train_images_digits, shuffle=True)
#%%
trainer= dl.Trainer(max_epochs=1, accelerator="auto")
#%%
trainer.fit(classifier, train_dataloader)

#%%