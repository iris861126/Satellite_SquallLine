#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting for the finer grid resolution
"""
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cluster_2d import box_squall, pts
from setting import lat_s, lat_e, lon_s, lon_e, lat, lon
from setting import mask_domain
from setting import connectpoints, connectpoints_1
from setting import test_point
from setting import regions, num, i
import copy
# cartopy-0.17.0 pyshp-2.1.0

# load data
ws     = np.load('ws_{0}.npy'.format(num))
ws_all = np.load('ws_all_{0}.npy'.format(num))
ws = np.where(ws==-127,np.nan,ws)
ws = ws.transpose((0,2,1))
ws_all_2 = np.zeros((11,42))
ws_all_2[0:10,:] = ws_all 

if   num == 1:
    ntimes = int(24*8+7-1)
elif num == 2:
    ntimes = int(5+1-1)
elif num == 3:
    ntimes = int(23*8+1-1)
elif num == 4:
    ntimes = int(20*8+1-1)
elif num == 5:
    ntimes = int(19*8+3-1)
elif num == 6:
    ntimes = int(17*8+7-1)
elif num == 7:
    ntimes = int(21*8+2-1)
elif num == 8:
    ntimes = int(17*8+8-1)
elif num == 9:
    ntimes = int(28*8+3-1)
elif num == 10:
    ntimes = int(16*8+1-1)    
    
ws_1 = ws[ntimes:ntimes+4]
temp = np.repeat(ws_1[:,:,:],10,axis=2)
ws_finer = np.repeat(temp,10,axis=1)

# domain for each case
mask_lat = mask_domain(num)[0]
mask_lon = mask_domain(num)[1]
mask_lat = np.equal(mask_lat,0)
mask_lon = np.equal(mask_lon,0)

temp = ws_finer[:,mask_lat,:]
ws_finer = temp[:,:,mask_lon]
lat_domain = lat[mask_lat]
lon_domain = lon[mask_lon]
lonGrid_do, latGrid_do = np.meshgrid(lon_domain, lat_domain)

# Classify 11 WSs into 3 categories
# class 1
test = np.where(ws_finer == 1,1,ws_finer)
test = np.where(test == 2,1,test)
ws_finer = np.where(test == 3,1,test)

# class 2
test = np.where(ws_finer == 4,2,ws_finer)
test = np.where(test == 5,2,test)
ws_finer = np.where(test == 6,2,test)

# class 3
test = np.where(ws_finer == 7,3,ws_finer)
test = np.where(test == 8,3,test)
test = np.where(test == 9,3,test)
ws_finer = np.where(test == 10,3,test)

# let WS11 & nan as nothing
ws_finer = np.where(ws_finer == 11,np.nan,ws_finer)

#%% Plot b:the distrubution of WS in each domain
"""
cm = (lon_s[num-1]+lon_e[num-1])/2
dlon = np.abs(lon_s[num-1]-cm)
proj = ccrs.PlateCarree(central_longitude=cm)

cmap = copy.copy(mpl.cm.get_cmap("Paired"))
cmap.set_bad(color='white')
mycolors = cmap
bounds = np.arange(0.5,4.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds,mycolors.N)

fig = plt.figure(figsize=[10, 8])
ax = fig.add_subplot(1, 1, 1, projection=proj)
plt.subplots_adjust(top=0.958,
                    bottom=0.019,
                    left=0.019,
                    right=0.981,
                    hspace=0.0,
                    wspace=0.0)

im = plt.pcolormesh(lon_domain, 
               lat_domain, 
               ws_finer[i,:,:], 
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
plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), 
             ax=ax,
             ticks=[1,2,3],
             orientation='horizontal',
             shrink=0.75,
             aspect=30)
# plot the subregiions
connectpoints(ax,box_squall[:,0],box_squall[:,1],0,1)
connectpoints(ax,box_squall[:,0],box_squall[:,1],1,2)
connectpoints(ax,box_squall[:,0],box_squall[:,1],2,3)
connectpoints(ax,box_squall[:,0],box_squall[:,1],3,0)

if num==8:
    # define the rear region & front region
    connectpoints_1(ax,pts[:,0],pts[:,1],0,1)
    connectpoints_1(ax,pts[:,0],pts[:,1],0,3)
    connectpoints_1(ax,pts[:,0],pts[:,1],3,2)
    connectpoints_1(ax,pts[:,0],pts[:,1],2,1)
else:   
    # define the rear region & front region
    connectpoints_1(ax,pts[:,0],pts[:,1],0,1)
    connectpoints_1(ax,pts[:,0],pts[:,1],1,3)
    connectpoints_1(ax,pts[:,0],pts[:,1],3,2)
    connectpoints_1(ax,pts[:,0],pts[:,1],2,0)

name = str(num)+"b"
plt.savefig(name+".png", dpi=1200)
plt.show()

name = str(num)+"b"
plt.savefig(name+"_cld.png", dpi=1200)
plt.show()
"""
#%% for squall line, rear and front region
vertices_s = [(box_squall[0,0],box_squall[0,1]),
              (box_squall[1,0],box_squall[1,1]),
              (box_squall[2,0],box_squall[2,1]),
              (box_squall[3,0],box_squall[3,1])]
if num == 8:
    vertices_r = [(box_squall[0,0],box_squall[0,1]),
                  (box_squall[1,0],box_squall[1,1]),
                  (pts[1,0],pts[1,1]),
                  (pts[0,0],pts[0,1])]
    vertices_f = [(box_squall[2,0],box_squall[2,1]),
                  (box_squall[3,0],box_squall[3,1]),
                  (pts[3,0],pts[3,1]),
                  (pts[2,0],pts[2,1])]
else:
    vertices_r = [(box_squall[0,0],box_squall[0,1]),
                  (box_squall[3,0],box_squall[3,1]),
                  (pts[1,0],pts[1,1]),
                  (pts[0,0],pts[0,1])]
    vertices_f = [(box_squall[1,0],box_squall[1,1]),
                  (box_squall[2,0],box_squall[2,1]),
                  (pts[3,0],pts[3,1]),
                  (pts[2,0],pts[2,1])]

all_shapes = [vertices_s, vertices_r, vertices_f]
ws_region   = np.zeros((3, np.size(lat_domain),np.size(lon_domain)))

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

    temp = ws_finer[i,:,:]
    temp = np.where(mask_region == 1,temp,np.nan)
    ws_region[l,:,:] = temp

np.save('ws_cld_regions_{0}.npy'.format(num),ws_region)

#%% the fraction of WSs in each subregion
#WS=[1,2,3,4,5,6,7,8,9,10,11]
WS=[1,2,3]
count = np.zeros((np.size(regions),np.size(WS)))
frac  = np.zeros((np.size(regions),np.size(WS)))
for area in range(np.size(regions)):
    temp = ws_region[area,:,:]
    for x in range(np.size(ws_region,axis=1)):
        for y in range(np.size(ws_region,axis=2)):
            for kind in range(np.size(WS)):
                if temp[x,y] == WS[kind]:
                    count[area,kind] += 1
                
    frac[area] = count[area]/np.sum(count[area])

np.savetxt('cld_count_{0}.txt'.format(num),count)
np.savetxt('cld_frac_{0}.txt'.format(num),frac)

"""
#%% Plot c: plot the squall line, rear and front region
cmap = copy.copy(mpl.cm.get_cmap("Paired"))
cmap.set_bad(color='white')
mycolors = cmap
bounds = np.arange(0.5,4.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds,mycolors.N)

cm = (lon_s[num-1]+lon_e[num-1])/2
dlon = np.abs(lon_s[num-1]-cm)
proj = ccrs.PlateCarree(central_longitude=cm)

fig = plt.figure(figsize=[13, 4])

plt.subplots_adjust(top=0.86,
                    bottom=0.014,
                    left=0.029,
                    right=0.971,
                    hspace=0.2,
                    wspace=0.176)

lon = np.arange(lon_s[num-1],lon_e[num-1],0.1)
lat = np.arange(lat_s[num-1]+0.1,lat_e[num-1]+0.1,0.1)

for s in range(3):
    ax = fig.add_subplot(1, 3, 1+s, projection=proj)
    im = plt.pcolormesh(lon_domain, 
                        lat_domain, 
                        ws_region[s,:,:],
                        cmap=mycolors, norm=norm,
                        transform=ccrs.PlateCarree())

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
    ax.gridlines(draw_labels=True , crs=ccrs.PlateCarree(), xlocs=xlocs, ylocs=ylocs)

    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), 
                 ax=ax,
                 ticks=[1,2,3],
                 orientation='horizontal',
                 shrink=0.75,
                 aspect=30)
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
    
    ax.set_title("Subregion: "+regions[s], fontsize='9',fontweight='bold')

 
name = str(num)+"c"
plt.savefig(name+".png", dpi=1200)
plt.show()

name = str(num)+"c"
plt.savefig(name+"_cld.png", dpi=1200)
plt.show()
#%% Plot WS: plot 11 ws

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
"""
