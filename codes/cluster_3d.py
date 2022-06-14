#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSs clusters(3d, including time dimension)
"""
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.img_tiles as cimgt
from setting import lat_s, lat_e, lon_s, lon_e
from setting import mask_domain, mask_cloud

#%% change the number to run different cases (1~10)
num = 1

#%% load data
itmp = np.load('itmp_{0}.npy'.format(num))
cloud = np.load('cloud_{0}.npy'.format(num))
lat = np.genfromtxt('lat.txt')
lon = np.genfromtxt('lon.txt')
tmptab = np.genfromtxt('tmptab.txt')

#%% derive the TB values
itmp_int = itmp.astype(int)
mask = np.equal(itmp_int,-9223372036854775808)
itmp[~mask] = tmptab[itmp_int[~mask]]

#%% apply the cloud_mask
cloud_mask = mask_cloud(cloud)[0]
itmp = np.where(cloud_mask==1,itmp,np.nan)

#%% domain for each case
mask_lat = mask_domain(num)[0]
mask_lon = mask_domain(num)[1]
mask_lat = np.equal(mask_lat,0)
mask_lon = np.equal(mask_lon,0)

itmp_1 = itmp[:,mask_lat,:]
itmp_2 = itmp_1[:,:,mask_lon]

lat_domain = lat[mask_lat]
lon_domain = lon[mask_lon]
lonGrid_do, latGrid_do = np.meshgrid(lon_domain, lat_domain)

cloud_TB = itmp_2

#%% generate 3D clusters 
labels = measure.label(cloud_TB, connectivity=3) 
properties = measure.regionprops(labels)
centroid_time = np.zeros((np.max(labels)))
centroid_lats = np.zeros((np.max(labels)))
centroid_lons = np.zeros((np.max(labels)))
label = np.zeros((np.max(labels)))

i = 0
for prop in properties:
    centroid_time[i] = prop.centroid[0]
    centroid_lats[i] = prop.centroid[1]
    centroid_lons[i] = prop.centroid[2]
    label[i] = prop.label
    i += 1

centroid_time = centroid_time.astype(int)
centroid_lats = centroid_lats.astype(int)
centroid_lons = centroid_lons.astype(int)

labels_nan = np.where(labels == 1, np.nan, labels)

#%% plot
current_cmap = matplotlib.cm.get_cmap('hsv')
current_cmap.set_bad(color='white')
fig, axes = plt.subplots(3,8,constrained_layout=True)

i=0
while (i<24):
    ax = plt.subplot(3,8,1+i, projection=ccrs.PlateCarree())
    box = [lon_s[num-1], lon_e[num-1], lat_s[num-1], lat_e[num-1]]
    ax.set_extent(box,crs=ccrs.PlateCarree())
    """
    ax.set_xticks(np.arange(118,123.5+0.5, 0.5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(22.5, 28+0.5, 0.5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    """
    # terrain
    stamen_terrain = cimgt.StamenTerrain()
    ax.add_image(stamen_terrain,8)

    ax.pcolor(lon_domain,lat_domain,labels_nan[i,:,:],cmap=current_cmap)
    ax.coastlines('10m')
    
    i += 1

