# -*- coding: utf-8 -*-
"""
Put czi file into folder with same name
"""
#%%
import os
import tkinter as tk
from tkinter.filedialog import askdirectory
#%%
path_root= askdirectory(title='Select Folder for CZI folder') # shows dialog box and return the path
#path_root="C:/Users/BioMecaCell/Documents/Eloise/Raw/24-04-17-FucciHela-100x-4.5fibrostrep-constantfield"
old_path=askdirectory(title='Select Folder of Data')
new_path=os.path.join(path_root,"CZI")
os.mkdir(new_path)
list_files=os.listdir(old_path)
for file in list_files :
    new_path_file=os.path.join(new_path,file[:-4])
    os.mkdir(new_path_file)
    os.rename(os.path.join(old_path,file), os.path.join(new_path_file,file))


#%%
path_root=askdirectory(title='Select Folder for images folder')
old_path=askdirectory(title='Select Folder of Data')
new_path=os.path.join(path_root,"P4")
os.mkdir(new_path)
list_files=os.listdir(old_path)
for file in list_files :
    new_path_file=os.path.join(new_path,file[:-4])
    os.mkdir(new_path_file)
    os.rename(os.path.join(old_path,file), os.path.join(new_path_file,file))