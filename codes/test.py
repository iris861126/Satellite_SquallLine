#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: the histogram
"""
import matplotlib.pyplot as plt
import numpy as np
from setting import regions
from setting import survey,cloud_type_list, ws_list
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.patches as  mpatches
from scipy.interpolate import make_interp_spline

ws_count = np.zeros((10,3,9))
ws_frac = np.zeros((10,3,9))

for kk in range(10):
    num = kk+1
    #ws_count[kk,:] = np.genfromtxt('count_{0}.txt'.format(num))
    ws_count[kk,:] = np.genfromtxt('count_cldamt_{0}.txt'.format(num))
    #ws_count[kk,:] = np.genfromtxt('cld_count_{0}.txt'.format(num))
    
    for l in range(3):
        ws_frac[kk,l,:] = ws_count[kk,l,:]/np.sum(ws_count[kk,l])
 
mean_ws_frac = np.mean(ws_frac,axis=0)
#%%
# for 3 subregions
colors=['skyblue',"salmon","olive"]
colors_area=['aliceblue',"mistyrose","palegoldenrod"]

fig = plt.figure(figsize=(30, 10))
for l in range(3):
    ax = fig.add_subplot(1, 3, 1+l)
    ax.set_title("{0} Subregion".format(regions[l]),fontsize=13,fontweight='bold')
    plt.subplots_adjust(top=0.95,
                        bottom=0.119,
                        left=0.035,
                        right=0.992,
                        hspace=0.1,
                        wspace=0.102)
    
    for k in range(10):
        x = np.array([1,2,3,4,5,6,7,8,9])
        #x = np.array([1,2,3])
        y = ws_frac[k,l]
        model=make_interp_spline(x,y)
        xs=np.linspace(1,9,500)
        ys=model(xs)
        #ax.plot(x, y,color=colors_area[l],linewidth=1.5)
        ax.plot(xs, ys,color=colors_area[l],linewidth=1.5)

    x = np.array([1,2,3,4,5,6,7,8,9])
    #x = np.array([1,2,3])
    y = mean_ws_frac[l]
    model=make_interp_spline(x,y)
    xs=np.linspace(1,9,500)
    ys=model(xs)
    ax.plot(xs, ys,color=colors[l],linewidth=4.0,label=regions[l])
    plt.legend()

    ax.set_xlim((0.7, 9.3))
    plt.xticks(np.arange(1,10,1))
    #labels = ['C','I','S']
    #labels = ws_list
    labels = cloud_type_list
    ax.set_xticklabels(labels)

    if l == 1:
        #ax.set_xlabel("WSs",fontsize=12,fontweight='bold')
        ax.set_xlabel("Cloud Types",fontsize=12,fontweight='bold')
        #ax.set_xlabel("Classified WSs",fontsize=12,fontweight='bold')

    if l == 0:
        ax.set_ylabel("Fraction [%]",fontsize=12,fontweight='bold')
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)        

#plt.savefig("3_mode.png".format(num), dpi=1200)
plt.savefig("cld_all.png", dpi=1200)

plt.show()
#%%
"""
for kk in range(10):
    num = kk+1
    cloud_region  = np.load('cloud_region_{0}.npy'.format(num))
    count  = np.genfromtxt('count_cldamt_{0}.txt'.format(num))
    frac  = np.zeros((3,9))

    colors=['skyblue',"salmon","olive"]
    for l in range(3):
        for kind in range(9):
            count[l,kind]=np.sum(cloud_region[l,kind,:,:]) 
    
        frac[l] = count[l]/np.sum(count[l])
        frac_total[kk]=frac

        x = np.array([1,2,3,4,5,6,7,8,9])
        y = frac[l]

        model=make_interp_spline(x,y)

        xs=np.linspace(1,9,500)
        ys=model(xs)
        ax.plot(xs, ys,color=colors[l],label=regions[l])

        ax.set_xlim((0, 10))
        plt.xticks(np.arange(1,10,1))
        
    ax.set_xlabel("Cloud Types")
    ax.set_ylabel("Fraction [%]")
    labels = cloud_type_list
    ax.set_xticklabels(labels)
    #plt.legend()
        
    #plt.savefig("cld_frac_{0}.png".format(num), dpi=1200)
    #plt.show()
    
    
    
    
    
    
    
    s = ws_init[0].reshape(np.size(ws_init,axis=1)*np.size(ws_init,axis=2))
    r = ws_init[1].reshape(np.size(ws_init,axis=1)*np.size(ws_init,axis=2))
    f = ws_init[2].reshape(np.size(ws_init,axis=1)*np.size(ws_init,axis=2))
    ms = np.ma.masked_invalid(s).mask
    mr = np.ma.masked_invalid(r).mask
    mf = np.ma.masked_invalid(f).mask
    s = s[~ms]
    r = r[~mr]
    f = f[~mf]
    
    region_list = [s,r,f]
    
    df = pd.DataFrame(region_list[0], columns=["value"])
    df1 = pd.DataFrame(region_list[1], columns=["value"])
    df2 = pd.DataFrame(region_list[2], columns=["value"])
    
    fig = plt.figure(figsize=(25.4, 20))
    
    plt.subplots_adjust(top=0.981,
                        bottom=0.057,
                        left=0.062,
                        right=0.987,
                        hspace=0.175,
                        wspace=0.2)
    ax = fig.add_subplot(3, 1, 1)
    plt.hist([df.value, df1.value, df2.value], 
             bins=np.arange(11+1)-0.5, 
             align='mid', 
             label=regions, 
             stacked=False, 
             color=['skyblue',"salmon","olive"], 
             alpha=0.6,
             density=True)
    ax.set_xlim((0, 12))
    plt.xticks(np.arange(1,12,1))
    labels = ['WS1', 'WS2','WS3','WS4', 'WS5','WS6','WS7', 'WS8','WS9','WS10', 'WS11']
    ax.set_xticklabels(labels)
    plt.legend()
    
    
    ax = fig.add_subplot(3, 1, 2)
    kde = stats.gaussian_kde(df.value)
    kde1 = stats.gaussian_kde(df1.value)
    kde2 = stats.gaussian_kde(df2.value)
    xx = np.linspace(1, 11, 1000)
    ax.plot(xx, kde(xx),color='skyblue',label=regions[0])
    ax.plot(xx, kde1(xx),color='salmon',label=regions[1])
    ax.plot(xx, kde2(xx),color='olive',label=regions[2])
    ax.set_xlim((0, 12))
    plt.xticks(np.arange(1,12,1))
    labels = ['WS1', 'WS2','WS3','WS4', 'WS5','WS6','WS7', 'WS8','WS9','WS10', 'WS11']
    ax.set_xticklabels(labels)
    plt.legend()
    
    
    category_names = labels
    ws_count = np.genfromtxt('count_{0}.txt'.format(num))
    ws_frac = np.genfromtxt('frac_{0}.txt'.format(num))
    results = {
        regions[0]: ws_frac[0],
        regions[1]: ws_frac[1],
        regions[2]: ws_frac[2]}
    survey(fig, results, category_names)
    plt.savefig("WS_frac_{0}.png".format(num), dpi=1200)
    plt.show()
"""
#%% for 9 cloud types
"""
count = np.zeros((10,3,9))
frac = np.zeros((10,3,9))

for kk in range(10):
    num = kk+1
    count[kk]  = np.genfromtxt('count_cldamt_{0}.txt'.format(num))    
    for l in range(3):
        frac[kk,l,:] = count[kk,l,:]/np.sum(count[kk,l])
 
mean_frac = np.mean(frac,axis=0)
# for 3 subregions
colors=['skyblue',"salmon","olive"]
colors_area=['aliceblue',"mistyrose","palegoldenrod"]

fig = plt.figure(figsize=(30, 10))
for l in range(3):
    ax = fig.add_subplot(1, 3, 1+l)
    ax.set_title("Composition of 9 Cloud Types in {0} Subregion".format(regions[l]),fontsize=13,fontweight='bold')
    plt.subplots_adjust(top=0.95,
                        bottom=0.119,
                        left=0.035,
                        right=0.992,
                        hspace=0.1,
                        wspace=0.102)
    
    for k in range(10):
        x = np.array([1,2,3,4,5,6,7,8,9])
        y = frac[k,l]
        model=make_interp_spline(x,y)
        xs=np.linspace(1,9,500)
        ys=model(xs)
        ax.plot(xs, ys,color=colors_area[l],linewidth=1.5)
        
    x = np.array([1,2,3,4,5,6,7,8,9])
    y = mean_frac[l]
    model=make_interp_spline(x,y)
    xs=np.linspace(1,9,500)
    ys=model(xs)
    ax.plot(xs, ys,color=colors[l],linewidth=4.0,label=regions[l])
    plt.legend()

    ax.set_xlim((0.7, 9.3))
    plt.xticks(np.arange(1,10,1))
    labels = cloud_type_list
    ax.set_xticklabels(labels)

    if l == 1:
        ax.set_xlabel("Cloud Types",fontsize=12,fontweight='bold')
    if l == 0:
        ax.set_ylabel("Fraction [%]",fontsize=12,fontweight='bold')
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)        

plt.savefig("cld_frac_{0}.png".format(num), dpi=1200)
plt.show()


for kk in range(10):
    num = kk+1
    cloud_region  = np.load('cloud_region_{0}.npy'.format(num))
    count  = np.genfromtxt('count_cldamt_{0}.txt'.format(num))
    frac  = np.zeros((3,9))

    colors=['skyblue',"salmon","olive"]
    for l in range(3):
        for kind in range(9):
            count[l,kind]=np.sum(cloud_region[l,kind,:,:]) 
    
        frac[l] = count[l]/np.sum(count[l])
        frac_total[kk]=frac

        x = np.array([1,2,3,4,5,6,7,8,9])
        y = frac[l]

        model=make_interp_spline(x,y)

        xs=np.linspace(1,9,500)
        ys=model(xs)
        ax.plot(xs, ys,color=colors[l],label=regions[l])

        ax.set_xlim((0, 10))
        plt.xticks(np.arange(1,10,1))
        
    ax.set_xlabel("Cloud Types")
    ax.set_ylabel("Fraction [%]")
    labels = cloud_type_list
    ax.set_xticklabels(labels)
    #plt.legend()
        
    #plt.savefig("cld_frac_{0}.png".format(num), dpi=1200)
    #plt.show()
"""