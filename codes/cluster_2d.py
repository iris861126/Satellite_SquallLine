#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: CSs clusters (work with TB)
"""
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from setting import lat_s, lat_e, lon_s, lon_e
from setting import slat_s, slat_e, slon_s, slon_e
from setting import mask_domain, connectpoints, connectpoints_1
from setting import num, i
import cv2
import copy

#%% load data
itmp = np.load('it_{0}.npy'.format(num))
cloud = np.load('cld_{0}.npy'.format(num))
lat = np.genfromtxt('lat.txt')
lon = np.genfromtxt('lon.txt')
tmptab = np.genfromtxt('tmptab.txt')

# derive the TB values
itmp_int = itmp.astype(int)
mask = np.equal(itmp_int,-9223372036854775808)
itmp[~mask] = tmptab[itmp_int[~mask]]

# apply the cloud_mask
cloud_mask = np.where(cloud==1,cloud,0)
itmp = np.where(cloud_mask==1,itmp,np.nan)

# domain for each case
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
np.save('TB_{0}.npy'.format(num),cloud_TB)

#%% define the squall line region
    
points = cloud_TB[i,:,:]

# TB < 230K, except case 1 [TB < 245K] & case 2 [TB < 225K]
if num == 1:
    squall = np.where(points <= 245,points,-1000)
elif num == 2:
    squall = np.where(points <= 225,points,-1000)
else:
    squall = np.where(points <= 230,points,-1000)

mask_squall = np.equal(squall,-1000)
squall_lons = lonGrid_do[~mask_squall]
squall_lats = latGrid_do[~mask_squall]

cnt = np.array([squall_lons,squall_lats])
cnt = np.where((((cnt[0]<=slon_e[num-1]) & (cnt[0]>=slon_s[num-1])) & ((cnt[1]<=slat_e[num-1]) & (cnt[1]>=slat_s[num-1]))),cnt,-1000)
mask_cnt = np.equal(cnt,-1000)
cnt= cnt[~mask_cnt]
cnt = np.reshape(cnt,(2,int(np.size(cnt)/2)))
cnt = np.array(cnt, dtype=np.float32)
cnt = np.transpose(cnt,(1,0))
rect = cv2.minAreaRect(cnt) 
box_squall = cv2.boxPoints(rect)

theta = rect[2]

a = rect[1][0]
b = rect[1][1]
if a > b:
    L = a
    W = b
else:
    L = b
    W = a

#%% Plot a
fig = plt.figure(figsize=[10, 8])
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
plt.subplots_adjust(top=0.958,
                    bottom=0.019,
                    left=0.019,
                    right=0.981,
                    hspace=0.0,
                    wspace=0.0)
connectpoints(ax,box_squall[:,0],box_squall[:,1],0,1)
connectpoints(ax,box_squall[:,0],box_squall[:,1],1,2)
connectpoints(ax,box_squall[:,0],box_squall[:,1],2,3)
connectpoints(ax,box_squall[:,0],box_squall[:,1],3,0)

rf_x = np.zeros((4)) 
rf_y = np.zeros((4))
if num==8:
    # define the rear region & front region
    dx = np.abs(box_squall[0,0]-box_squall[3,0])
    dy = np.abs(box_squall[0,1]-box_squall[3,1])
    rf_x[0] = box_squall[0,0]-2*dx
    rf_y[0] = box_squall[0,1]-2*dy
    rf_x[1] = box_squall[1,0]-2*dx
    rf_y[1] = box_squall[1,1]-2*dy
    
    rf_x[2] = box_squall[2,0]+dx
    rf_y[2] = box_squall[2,1]+dy
    rf_x[3] = box_squall[3,0]+dx
    rf_y[3] = box_squall[3,1]+dy
    pts = np.array([rf_x,rf_y]).transpose()
    
    connectpoints_1(ax,pts[:,0],pts[:,1],0,1)
    connectpoints_1(ax,pts[:,0],pts[:,1],0,3)
    connectpoints_1(ax,pts[:,0],pts[:,1],3,2)
    connectpoints_1(ax,pts[:,0],pts[:,1],2,1)
else:   
    # define the rear region & front region
    dx = np.abs(box_squall[0,0]-box_squall[1,0])
    dy = np.abs(box_squall[0,1]-box_squall[1,1])
    rf_x[0] = box_squall[0,0]-2*dx
    rf_y[0] = box_squall[0,1]+2*dy
    rf_x[1] = box_squall[3,0]-2*dx
    rf_y[1] = box_squall[3,1]+2*dy
    
    rf_x[2] = box_squall[1,0]+dx
    rf_y[2] = box_squall[1,1]-dy
    rf_x[3] = box_squall[2,0]+dx
    rf_y[3] = box_squall[2,1]-dy
    pts = np.array([rf_x,rf_y]).transpose()

    connectpoints_1(ax,pts[:,0],pts[:,1],0,1)
    connectpoints_1(ax,pts[:,0],pts[:,1],1,3)
    connectpoints_1(ax,pts[:,0],pts[:,1],3,2)
    connectpoints_1(ax,pts[:,0],pts[:,1],2,0)


cmap = copy.copy(mpl.cm.get_cmap("jet_r"))
cmap.set_bad(color='white')
mycolors = cmap
bounds = np.arange(200,320,10)
norm = matplotlib.colors.BoundaryNorm(bounds,mycolors.N)

labels_nan = cloud_TB[i,:,:] 
box = [lon_s[num-1], lon_e[num-1], lat_s[num-1], lat_e[num-1]]
ax.set_extent(box,crs=ccrs.PlateCarree())

ax.coastlines('10m')
plt.pcolormesh(lon_domain, 
               lat_domain, 
               labels_nan, 
               cmap=mycolors, norm=norm,
               transform=ccrs.PlateCarree())

plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), 
             ax=ax,
             ticks=bounds,
             orientation='horizontal',
             shrink=0.75,
             aspect=30)
xlocs = np.arange(lon_s[num-1],lon_e[num-1],1) 
ylocs = np.arange(lat_s[num-1],lat_e[num-1],1) 
if num == 5 or num == 8 or num == 9 or num == 10:
    xlocs = xlocs-360        

ax.gridlines(draw_labels=True , crs=ccrs.PlateCarree(), xlocs=xlocs, ylocs=ylocs)
"""
name = str(num)+"a"
plt.savefig(name+".png", dpi=1200)
plt.show()
"""