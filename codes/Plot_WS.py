#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting
"""
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cluster_2d import num, i, rf_x, rf_y, box_squall, theta
from setting import lat_s, lat_e, lon_s, lon_e, lat_ws, lon_ws
from setting import mask_domain_ws
from setting import connectpoints, connectpoints_1
import copy
# cartopy-0.17.0 pyshp-2.1.0

#%% load data
ws     = np.load('ws_{0}.npy'.format(num))
ws_all = np.load('ws_all_{0}.npy'.format(num))

ws = np.where(ws==-127,np.nan,ws)
ws = ws.transpose((0,2,1))
temp = np.repeat(ws[:,:,:],10,axis=2)
ws_finer = np.repeat(temp,10,axis=1)

ws_all_2 = np.zeros((11,42))
ws_all_2[0:10,:] = ws_all 

# domain for each case
mask_lat = mask_domain_ws(num)[0]
mask_lon = mask_domain_ws(num)[1]
mask_lat = np.equal(mask_lat,0)
mask_lon = np.equal(mask_lon,0)

ws_1 = ws[:,mask_lat,:]
ws_2 = ws_1[:,:,mask_lon]

if num == 1:
    ntimes = int(24*8+7-1)
elif num == 2:
    ntimes = int(23*8+1-1)
elif num == 3:
    ntimes = int(23*8+1-1)
elif num == 4:
    ntimes = int(19*8+7-1)
elif num == 5:
    ntimes = int(19*8+3-1)
elif num == 6:
    ntimes = int(18*8+7-1)
elif num == 7:
    ntimes = int(21*8+2-1)
elif num == 8:
    ntimes = int(18*8+0-1)
elif num == 9:
    ntimes = int(28*8+3-1)
elif num == 10:
    ntimes = int(16*8+0-1)    
    
ws_f = ws_2[ntimes:ntimes+4]

lat_domain = lat_ws[mask_lat]
lon_domain = lon_ws[mask_lon]
lonGrid_do, latGrid_do = np.meshgrid(lon_domain, lat_domain)

#%% Plot2: plot the distrubution of ws
cm = (lon_s[num-1]+lon_e[num-1])/2
dlon = np.abs(lon_s[num-1]-cm)
proj = ccrs.PlateCarree(central_longitude=cm)

cmap = copy.copy(mpl.cm.get_cmap("Paired"))
cmap.set_bad(color='white')
mycolors = cmap
bounds = np.arange(0.5,12.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds,mycolors.N)

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(1, 1, 1, projection=proj)
plt.subplots_adjust(top=0.958,
                    bottom=0.019,
                    left=0.019,
                    right=0.981,
                    hspace=0.0,
                    wspace=0.0)

im = plt.pcolormesh(lon_domain, 
                   lat_domain, 
                   ws_f[i,:,:], 
                   cmap=mycolors, norm=norm,
                   transform=ccrs.PlateCarree())

minlon = -dlon + cm
maxlon = +dlon + cm
minlat = lat_s[num-1]
maxlat = lat_e[num-1]

ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

xlocs = np.arange(minlon,maxlon,1) 
ylocs = np.arange(minlat,maxlat,1) 

if num == 5 or num == 8 or num == 9 or num == 10:
    xlocs = xlocs-360        

ax.coastlines('10m')
ax.gridlines(draw_labels=True , crs=ccrs.PlateCarree(), xlocs=xlocs, ylocs=ylocs)
#colorbar_index(ncolors=11, cmap=cmap)    
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), 
             ax=ax,
             ticks=[1,2,3,4,5,6,7,8,9,10,11],
             orientation='horizontal',
             shrink=0.75,
             aspect=30)
# plot the subregiions
connectpoints(ax,box_squall[:,0],box_squall[:,1],0,1)
connectpoints(ax,box_squall[:,0],box_squall[:,1],1,2)
connectpoints(ax,box_squall[:,0],box_squall[:,1],2,3)
connectpoints(ax,box_squall[:,0],box_squall[:,1],3,0)

if num == 1 or num == 2 or num == 3 or num == 4 or num == 5 or num == 6 or num == 7 or num == 9 or num == 10:   
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

elif num==8:
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

#%% Plot3: plot 11 ws
levpc = np.genfromtxt("levpc.txt")
levtau = np.genfromtxt("levtau.txt")
levtau = levtau.round(2)
cmap = matplotlib.colors.ListedColormap(['#ffffff','#261585','#3b18e4',
                                         '#4869cc','#6b9164','#297b16',
                                         '#3fbb3b','#47e418','#f7d600','#fbff00']) 
bounds = np.array([0., 0.2, 1., 2., 3., 4., 6., 8., 10., 15., 99.])
norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N)
fig, axes = plt.subplots(nrows=4, ncols=3,constrained_layout=True,figsize=[10, 10],dpi=100)

k = 0
for ax in axes.flat:
    if k != 11:
        ax.imshow(100*ws_all_2[k,:].reshape(6,7).T,cmap=cmap,norm=norm, aspect='auto')
        ax.axvline(x=1.5,linewidth=2, color=(254/255, 222/255, 213/255, 0.65))
        ax.axhline(y=2.5,linewidth=2, color=(254/255, 222/255, 213/255, 0.65))
        ax.axvline(x=3.5,linewidth=2, color=(254/255, 222/255, 213/255, 0.65))
        ax.axhline(y=4.5,linewidth=2, color=(254/255, 222/255, 213/255, 0.65))
        ax.set_xticks(np.arange(len(levtau)))
        ax.set_yticks(np.arange(len(levpc)))
        ax.set_xticklabels(levtau,fontsize='9')
        ax.set_yticklabels(levpc,fontsize='9')
        ax.set_xlabel('τ',fontsize='9', fontweight='bold')
        ax.set_ylabel('CTP [hPa]',fontsize='9', fontweight='bold')
        ax.set_title('WS'+str(k+1),fontsize='10', fontweight='bold')
    k += 1
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), 
                    ax=axes, 
                    ticks = bounds,
                    orientation='horizontal',
                    extend='max',
                    aspect=60)


plt.savefig("WS.png", dpi=1200)
plt.show()
