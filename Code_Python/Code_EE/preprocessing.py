# -*- coding: utf-8 -*-
"""
EE
Preprocessing for confocal experiments
Get Field file with timestamps
"""

import numpy as np
import os
from datetime import datetime

#date="17_04_24"
#exp="17_04_24_ConstantField"
files_path="C:/Users/BioMecaCell/Documents/Eloise/Raw/24-04-24-fuccihela-constantfield-100x-4.5fibrostep/CZI/P2"
#files_path="C:/Users/BioMecaCell/Documents/Eloise/Raw/"+date
#field_file=open(str(date+exp+"_Field"),'w')
#file_czi="23-12-06_M1_P1_C2-01.czi"
all_files = os.listdir(files_path)

field_value=5.0
#field_file=np.array()
datetime_init=datetime.utcfromtimestamp(0)
for file in all_files :
    files_czi=os.path.join(files_path,file)
    with open(os.path.join(files_czi,file + ".czi"), errors="ignore") as frame:
        print(file)
        metadata = str(frame.readlines())
        timedata = "<METADATA><Tags><AcquisitionTime>"
        timestamps=np.array(metadata.split(timedata)[1:])
        times=np.array([timestamps[i][0:26] for i in range(len(timestamps))])
        times=np.array([datetime.strptime(times[i],"%Y-%m-%dT%H:%M:%S.%f")-datetime_init for i in range(len(times))])
        times=np.array([float(time.total_seconds()*1000) for time in times])
        
        posdata = "</DetectorState><FocusPosition>"
        positions=np.array(metadata.split(posdata)[1:])
        pos=np.array([positions[i][10:18] for i in range(len(positions))])
        
        
        field_file=np.full((len(times),4),field_value)
        field_file[:,1]=times
        field_file[:,3]=pos
        np.savetxt(str(files_path+'/'+file+'_Field.txt'),field_file,delimiter='\t',fmt="%1.3f")
        

