# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:51:32 2024

@author: EE

"""

#%% import
import numpy as np
import os
from datetime import datetime
import pandas as pd
import re

#%% get file

#date="17_04_24"
#exp="17_04_24_ConstantField"
files_path="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo"
#files_path="C:/Users/BioMecaCell/Documents/Eloise/Raw/"+date
#field_file=open(str(date+exp+"_Field"),'w')
#file_czi="23-12-06_M1_P1_C2-01.czi"
all_files = os.listdir(files_path)


#%% get metadata

#datetime_init=datetime.utcfromtimestamp(0)
#for file in all_files :
    
    
#files_czi=os.path.join(files_path,file)
file=all_files[0]
with open(os.path.join(files_path,file), errors="ignore") as frame:
    print(file)
    metadata = str(frame.readlines())
    timedata = "<METADATA><Tags><AcquisitionTime>"
    timestamps=np.array(metadata.split(timedata)[1:])
    # times=np.array([timestamps[i][0:26] for i in range(len(timestamps))])
    # times=np.array([datetime.strptime(times[i],"%Y-%m-%dT%H:%M:%S.%f")-datetime_init for i in range(len(times))])
    # times=np.array([float(time.total_seconds()*1000) for time in times])
    
    posdata = "</DetectorState><FocusPosition>"
    positions=np.array(metadata.split(posdata)[1:])
    pos=np.array([positions[i][10:18] for i in range(len(positions))])
    
    exptimedata= "<METADATA><Tags><AcquisitionTime>"
    exptimes=np.array(metadata.split(exptimedata)[1:])
    exptime=np.array([exptimes[i].split('&lt;')[10] for i in range(len(exptimes))])
    exptime=np.array([exptime[i][16:28] for i in range(len(exptime))])
    exptime=np.array([float(float(k)*10**(-6)) for k in exptime])
    print(exptime)
    
    channeldata='<Channel IsActivated="true"'
    channels=metadata.split(channeldata)[1:]
    #channel=channels.split('Name=')[:]
    for i in range(len(channels)) :
        if '<Fluor>EGFP</Fluor>' in channels[i] : 
            green=i
        if '<Fluor>DsRed</Fluor>' in channels[i] : 
            red=i
    
    i_phrase=r'<Intensity Status="Valid" IsActivated="true" Unit="%">' + '\d+'
    matches = re.search(i_phrase,metadata[:])
    intens=re.findall(i_phrase,metadata[:])
    #intens=metadata.split(intensitydata)
    # for i in range(len(intens)) :
    #     if 'DsRed' in intens[i] :
    #         print('red',i)
    #     if 'EGFP' in intens[i] :
    #         print('green',i)
    
    exp_cond=pd.DataFrame()
    exp_cond['EGFP']=[exptime[green]]
    exp_cond['DsRed']=[exptime[red]]

    
    
        # field_file=np.full((len(times),4),field_value)
        # field_file[:,1]=times
        # field_file[:,3]=pos
        # np.savetxt(str(files_path+'/'+file+'_Field.txt'),field_file,delimiter='\t',fmt="%1.3f")
        
#%% get exposure time and channel order
def exptimeChannel(file_path):
    with open(file_path, errors="ignore") as frame:
        metadata = str(frame.readlines())
        
        exptimedata= "<METADATA><Tags><AcquisitionTime>"
        exptimes=np.array(metadata.split(exptimedata)[1:])
        exptime=np.array([exptimes[i].split('&lt;')[10] for i in range(len(exptimes))])
        exptime=np.array([exptime[i][16:28] for i in range(len(exptime))])
        exptime=np.array([float(float(k)*10**(-6)) for k in exptime])
        exp_cond=pd.DataFrame()
        
        channeldata='<Channel IsActivated="true"'
        channels=metadata.split(channeldata)[1:]
        #channel=channels.split('Name=')[:]
        for i in range(len(channels)) :
            if '<Fluor>EGFP</Fluor>' in channels[i] : 
                green=i
                exp_cond['EGFP']=[exptime[green]]
            if '<Fluor>DsRed</Fluor>' in channels[i] : 
                red=i
                exp_cond['DsRed']=[exptime[red]]
            if '<Fluor>TL Brightfield</Fluor>' in channels[i] :
                bright=i
                exp_cond['Bright']=[exptime[bright]]
        
       
        
        
        return(exp_cond)
#%%
#directory="E:/Pelin/Imaging_Data/23.12.19_FluoroTest"
#all_files=[f for f in os.listdir(directory) if f.endswith('.czi')]
#%%
# cond=pd.DataFrame()
# cond.index=all_files
# # cond['EGFP']=all_files
# # cond['Red']=all_files
# for file in all_files:
#     exp_cond=exptimeChannel(os.path.join(directory,file))
#     cond.at[file,'EGFP exp']=exp_cond.at[0,'EGFP']
#     cond.at[file,'Red exp']=exp_cond.at[0,'DsRed']
#     cond.at[file,'Bright exp']=exp_cond.at[0,'Bright']
#%%
# file_info=[i for i in os.listdir(directory) if i.endswith('_info.txt')]
# #file_info=[i for i in all_files if i.endswith('_info.txt')]
# info=open(os.path.join(directory,file_info[0]))
# metadata_=info.readlines()

#%%
def metadatafromtxt(fileInfo_txt,dir_file):
    """
    get the parameters of the experiment from the _info.txt file associated to the .czi file
    
    Parameters
    ----------
    fileInfo_txt : type : .txt
                    file to retrieve the metadata from

    Returns
    -------
    infometadata : type : pandas.DataFrame 
                indices : channels of the czi file as indicated in the _info.txt
                columns : 'intensity' : intensity of laser for each channel in %
                          'exp time' : exposure time for each channel in ms

    """
    infometadata=pd.DataFrame()
    info=open(os.path.join(dir_file,fileInfo_txt))
    metadata='\n'.join(info.readlines())
    
    channel_phrase=r'Information\|Image\|Channel\|Fluor #' + r'[\d] = ' + r'(.*?)' + r'\n'
    findChannel=re.findall(channel_phrase,metadata[:])
    infometadata.index=[i for i in findChannel]
    
    intens_phrase=r'Information\|Image\|Channel\|Intensity #' + r'[\d] = ' + r'([\d\.]*)' + r' %'
    findI=re.findall(intens_phrase,metadata[:])
    infometadata['intensity']=[float(i) for i in findI]
    
    exptime_phrase=r'Information\|Image\|Channel\|ExposureTime #' + r'[\d] = '+ r'([\d\.]*)' + r'\n'
    findExp=re.findall(exptime_phrase,metadata[:])
    infometadata['exp time']=[float(float(i)*10**(-6)) for i in findExp]

    return infometadata
#%%
#meta=metadatafromtxt(file_info[1])







