#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting 9 cloud types
"""
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cluster_2d import box_squall, pts
from plot import vertices_s, vertices_r, vertices_f
from setting import lat_s, lat_e, lon_s, lon_e, lat, lon
from setting import mask_domain
from setting import connectpoints, connectpoints_1
from setting import test_point
from setting import regions, num, cloud_type_list, i
import copy
"""
00 Cumulus liquid
01 Stratocumulus liquid
02 Stratus liquid
03 Cumulus ice
04 Stratocumulus ice
05 Stratus ice
06 Altocumulus liquid
07 Altostratus liquid
08 Nimbostratus liquid
09 Altocumulus ice
10 Altostratus ice
11 Nimbostratus ice
12 Cirrus liquid
13 Cirrostratus liquid
14 Deep convective liquid
15 Cirrus ice
16 Cirrostratus ice
17 Deep convective ice

Cirrus: 12+15
Cirrostratus: 13+16
Deep convective: 14+17
Altocumulus: 06+09
Altostratus: 07+10
Nimbostratus: 08+11
Cumulus: 00+03
Stratocumulus: 01+04
Stratus: 02+05
"""
#%% load the data
cldamt_types  = np.load('cldamt_{0}.npy'.format(num))
cldamt_types = np.where(cldamt_types==32767,np.nan,cldamt_types)

temp = np.repeat(cldamt_types[:,:,:,:],10,axis=3)
cldamt_types_finer = np.repeat(temp,10,axis=2)

# domain for each case
mask_lat = mask_domain(num)[0]
mask_lon = mask_domain(num)[1]
mask_lat = np.equal(mask_lat,0)
mask_lon = np.equal(mask_lon,0)

temp = cldamt_types_finer[:,:,mask_lat,:]
cldamt_types_finer = temp[:,:,:,mask_lon]
lat_domain = lat[mask_lat]
lon_domain = lon[mask_lon]
lonGrid_do, latGrid_do = np.meshgrid(lon_domain, lat_domain)

# 9 cloud types
cloud = np.zeros((4,9,np.size(cldamt_types_finer,axis=2),np.size(cldamt_types_finer,axis=3)))

cloud[:,2,:,:] = cldamt_types_finer[:,12,:,:]+cldamt_types_finer[:,15,:,:]
cloud[:,1,:,:] = cldamt_types_finer[:,13,:,:]+cldamt_types_finer[:,16,:,:]
cloud[:,0,:,:] = cldamt_types_finer[:,14,:,:]+cldamt_types_finer[:,17,:,:]
cloud[:,5,:,:] = cldamt_types_finer[:, 6,:,:]+cldamt_types_finer[:, 9,:,:]
cloud[:,4,:,:] = cldamt_types_finer[:, 7,:,:]+cldamt_types_finer[:,10,:,:]
cloud[:,3,:,:] = cldamt_types_finer[:, 8,:,:]+cldamt_types_finer[:,11,:,:]
cloud[:,8,:,:] = cldamt_types_finer[:, 0,:,:]+cldamt_types_finer[:, 3,:,:]
cloud[:,7,:,:] = cldamt_types_finer[:, 1,:,:]+cldamt_types_finer[:, 4,:,:]
cloud[:,6,:,:] = cldamt_types_finer[:, 2,:,:]+cldamt_types_finer[:, 5,:,:]

#%% Plot: the cldamt for 9 cloud types in each domain
cm = (lon_s[num-1]+lon_e[num-1])/2
dlon = np.abs(lon_s[num-1]-cm)
proj = ccrs.PlateCarree(central_longitude=cm)

cmap = copy.copy(mpl.cm.get_cmap("YlOrRd"))
cmap.set_bad(color='white')
cmap.set_under('w')
mycolors = cmap
bounds = np.arange(1,100+1,1)
norm = matplotlib.colors.BoundaryNorm(bounds,mycolors.N)
fig = plt.figure(figsize=[15, 15],tight_layout=True)

for s in range(9):
    ax = fig.add_subplot(3, 3, 1+s, projection=proj)
    im = plt.pcolormesh(lon_domain, 
                        lat_domain, 
                        cloud[i,s,:,:], 
                        cmap=mycolors, 
                        norm=norm,
                        transform=ccrs.PlateCarree())
    if num == 5 or num == 9 or num == 10:    
        plt.subplots_adjust(top=0.97,
                            bottom=0.02,
                            left=0.05,
                            right=0.95,
                            hspace=0.15, 
                            wspace=-0.25)
    elif num == 8:
        plt.subplots_adjust(top=0.97,
                            bottom=0.02,
                            left=0.05,
                            right=0.95,
                            hspace=0.15, 
                            wspace=-0.15)
    else:
        plt.subplots_adjust(top=0.97,
                            bottom=0.02,
                            left=0.05,
                            right=0.95,
                            hspace=0.15, 
                            wspace=-0.5)
    minlon = -dlon + cm
    maxlon = +dlon + cm
    minlat = lat_s[num-1]
    maxlat = lat_e[num-1]
    
    ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
    
    # plot some lines on the plot
    xlocs = np.arange(minlon,maxlon,1) 
    ylocs = np.arange(minlat,maxlat,1) 
    
    if num == 5 or num == 8 or num == 9 or num == 10:
        xlocs = xlocs-360        
    
    ax.coastlines('10m')
    gl = ax.gridlines(draw_labels=True , crs=ccrs.PlateCarree(), xlocs=xlocs, ylocs=ylocs)
    gl.xlabels_top = False
    gl.ylabels_right = False

    # plot the subregiions
    connectpoints(ax,box_squall[:,0],box_squall[:,1],0,1)
    connectpoints(ax,box_squall[:,0],box_squall[:,1],1,2)
    connectpoints(ax,box_squall[:,0],box_squall[:,1],2,3)
    connectpoints(ax,box_squall[:,0],box_squall[:,1],3,0)
    
    if num==8:
        connectpoints_1(ax,pts[:,0],pts[:,1],0,1)
        connectpoints_1(ax,pts[:,0],pts[:,1],0,3)
        connectpoints_1(ax,pts[:,0],pts[:,1],3,2)
        connectpoints_1(ax,pts[:,0],pts[:,1],2,1)
    else:   
        connectpoints_1(ax,pts[:,0],pts[:,1],0,1)
        connectpoints_1(ax,pts[:,0],pts[:,1],1,3)
        connectpoints_1(ax,pts[:,0],pts[:,1],3,2)
        connectpoints_1(ax,pts[:,0],pts[:,1],2,0)
    
    ax.set_title(cloud_type_list[s], fontsize='10',fontweight='bold')
    
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), 
                 ax=ax,
                 ticks=[10,20,30,40,50,60,70,80,90,100],
                 orientation='vertical',
                 extend='min',
                 aspect=30)
      

name = str(num)+"_cldamt"
plt.savefig(name+".png", dpi=1200)
plt.show()

#%% for each subregion
all_shapes = [vertices_s, vertices_r, vertices_f]
cloud_region = np.zeros((3,9,np.size(lat_domain),np.size(lon_domain)))
count = np.zeros((np.size(regions),np.size(cloud_type_list)))
frac  = np.zeros((np.size(regions),np.size(cloud_type_list)))
# for vertices in all_shapes[s]:
for l in range(3):
    vertices = all_shapes[l]
    mask_region = np.zeros((np.size(lat_domain),np.size(lon_domain)))
    
    for m in range(np.size(lon_domain)):
        x = lon_domain[m]
        for n in range(np.size(lat_domain)):
            y = lat_domain[n]
            s = 0
            if test_point(x, y, vertices):
                mask_region[n,m] = 1

    temp = cloud[i,:,:,:]
    temp = np.where(mask_region == 1,temp,0)
    cloud_region[l,:,:,:] = temp

    for kind in range(np.size(cloud_type_list)):
        count[l,kind]=np.sum(cloud_region[l,kind,:,:])      
    
np.savetxt('count_cldamt_{0}.txt'.format(num),count)

    
    
