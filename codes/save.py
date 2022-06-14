#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final project: Save TB, cloud and WS data from HXG and HGGWS files
"""
import numpy as np
import netCDF4 as nc
from os import listdir
from os.path import isfile, isdir, join
years_list = ['1997.nc', '2006.nc', '2007.nc', '2008.nc',  
              '2012.nc', '2013.nc', '2014.nc', '2017.nc', 
              '2017.nc', '2017.nc']

months = ['november',
          'may',
          'april',
          'june',
          'march',
          'june',
          'may',
          'february',
          'march',
          'may']

days = [30,31,30,30,31,30,31,28,31,31]
time = [240,248,240,240,248,240,248,224,248,248]

# declare the vaiables
cldamt_types = np.zeros((40,18,180,360))    # for HGG

itmp         = np.zeros((40,1800,3600))     # for HXG
cloud        = np.zeros((40,1800,3600))     # for HXG
ws           = np.zeros((8*303,360,180))    # for HGGWS
ws_all       = np.zeros((10,10,42))         # for HGGWS

# HGG data
path = '/Users/joyuwu/碩一下/衛星/final_project/ISCCP_HGG_Basic'
files = listdir(path)
files.sort()
files = files[1:41]

i = 0
for f in files:
    fullpath = join(path, f)
    f = nc.Dataset(fullpath)    
    cldamt_types[i,:,:,:] = f.variables['cldamt_types'][:]
    i += 1
    
#%% save the cloudamt data from HGG
np.save('cldamt_1.npy' ,cldamt_types[  0: 4,:,:,:])
np.save('cldamt_2.npy' ,cldamt_types[  4: 8,:,:,:])
np.save('cldamt_3.npy' ,cldamt_types[  8:12,:,:,:])
np.save('cldamt_4.npy' ,cldamt_types[ 12:16,:,:,:])
np.save('cldamt_5.npy' ,cldamt_types[ 16:20,:,:,:])
np.save('cldamt_6.npy' ,cldamt_types[ 20:24,:,:,:])
np.save('cldamt_7.npy' ,cldamt_types[ 24:28,:,:,:])
np.save('cldamt_8.npy' ,cldamt_types[ 28:32,:,:,:])
np.save('cldamt_9.npy' ,cldamt_types[ 32:36,:,:,:])
np.save('cldamt_10.npy',cldamt_types[ 36:40,:,:,:])

#%% HXG data
path = '/Users/joyuwu/碩一下/衛星/final_project/HXG_data'
files = listdir(path)
files.sort()
files = files[1:41]

i = 0
for f in files:
    if i == 0 or i == 1 or i == 2 or i == 3:
        fullpath = join(path, f)
        f = nc.Dataset(fullpath)    
        cloud[i,:,:] = f.variables['cloud'][:]
        itmp[i,:,:]  = f.variables['itmp'][:]
    i += 1
  
mask_cloud = np.where(cloud==1,cloud,0)

# save the TB, cloud data from HXG
np.save('it_1.npy' ,itmp[  0: 4,:,:])
np.save('it_2.npy' ,itmp[  4: 8,:,:])
np.save('it_3.npy' ,itmp[  8:12,:,:])
np.save('it_4.npy' ,itmp[ 12:16,:,:])
np.save('it_5.npy' ,itmp[ 16:20,:,:])
np.save('it_6.npy' ,itmp[ 20:24,:,:])
np.save('it_7.npy' ,itmp[ 24:28,:,:])
np.save('it_8.npy' ,itmp[ 28:32,:,:])
np.save('it_9.npy' ,itmp[ 32:36,:,:])
np.save('it_10.npy',itmp[ 36:40,:,:])

np.save('cld_1.npy' ,mask_cloud[  0: 4,:,:])
np.save('cld_2.npy' ,mask_cloud[  4: 8,:,:])
np.save('cld_3.npy' ,mask_cloud[  8:12,:,:])
np.save('cld_4.npy' ,mask_cloud[ 12:16,:,:])
np.save('cld_5.npy' ,mask_cloud[ 16:20,:,:])
np.save('cld_6.npy' ,mask_cloud[ 20:24,:,:])
np.save('cld_7.npy' ,mask_cloud[ 24:28,:,:])
np.save('cld_8.npy' ,mask_cloud[ 28:32,:,:])
np.save('cld_9.npy' ,mask_cloud[ 32:36,:,:])
np.save('cld_10.npy',mask_cloud[ 36:40,:,:])

#%% WS data
path_2 = '/Users/joyuwu/碩一下/衛星/final_project/WS_data'
files_2 = listdir(path_2)
files_2.sort()

i = 0
for j in range(np.size(years_list)):
    while i < 241:
        year = years_list[j]
        for k in range(np.size(files_2)):
            if(year == files_2[k]):  
                print('j= '+str(j)+'; year= '+str(files_2[k]))
                f = str(files_2[k])
                fullpath_2 = join(path_2, f)
                f = nc.Dataset(fullpath_2)
                temp  = f.variables[months[j]][:] 
                temp1 = f.variables['ws'][:] 
                
        ws[i:i+int(time[j]),:,:] = temp
        ws_all[j,:,:] = temp1
        # save the WS data from HGGWS
        np.save('ws_{0}.npy'.format(j+1), ws[i:i+int(time[j]),:,:])
        np.save('ws_all_{0}.npy'.format(j+1), ws_all[j,:,:])
        i = i+int(time[j])