#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save TB, cloud and WS data from HXG and HGGWS files
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
for i in range(10):
  np.save('cldamt_{}.npy'.format(i+1) ,cldamt_types[(i*4):(i+1)*4, :, :, :])

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
for i in range(10):
  np.save('it_{}.npy'.format(i+1) ,itmp[(i*4):(i+1)*4, :, :])
  np.save('cld_{}.npy'.format(i+1) ,mask_cloud[(i*4):(i+1)*4, :, :])

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
