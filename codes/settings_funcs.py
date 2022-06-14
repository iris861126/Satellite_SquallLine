#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General setting and functions
"""
import numpy as np
import cartopy.crs as ccrs
import cv2
import matplotlib.pyplot as plt
#%% change the number to show different cases (1~10)
num = 10

#=====================================================================
# Cases Selected
#---------------------------------------
# [Time Period: 1997/11- 2017/05] 
# [Area: USA, China, Taiwan and Korea] 
# [Total Cases: 10  cases]
#---------------------------------------
# Cases List:
# 1997/11/26 ⇢ Time( 6- 9): 1997/11/25 1800UTC  - 1997/11/26 0300UTC
# 2006/05/01 ⇢ Time( 4- 7): 2006/05/01 1200-2100 UTC
# 2007/04/24 ⇢ Time(16-19): 2007/04/24 0000-0900UTC
# 2008/06/20 ⇢ Time(12-15): 2008/06/20 1200-2100UTC
# 2012/03/20 ⇢ Time(10-13): 2012/03/20 0600-1500 UTC
# 2013/06/18 ⇢ Time(14-17): 2013/06/18 1800UTC  - 2013/06/19 0300UTC
# 2014/05/22 ⇢ Time( 9-12): 2014/05/22 0300-1200UTC
# 2017/02/19 ⇢ Time( 8-11): 2017/02/19 0000-0900 UTC
# 2017/03/29 ⇢ Time(10-13): 2017/03/29 0600-1500 UTC
# 2017/05/16 ⇢ Time(16-19): 2017/05/17 0000-0900 UTC
#======================================================================
years  = ['1997', '2006', '2007', '2008', '2012', '2013', '2014', '2017', '2017', '2017']
months = ['11'  , '05'  , '04'  , '06'  , '03'  , '06'  , '05'  , '02'  , '03'  , '05'  ]
days   = ['26'  , '02'  , '23'  , '20'  , '20'  , '18'  , '22'  , '19'  , '29'  , '16'  ]
times  = ['00','03','06','09','12','15','18','21']

# set text in url
# 1997/11/26: CS0523805267
# 2006/05/02: CS0834108371
# 2007/04/24: CS0867608705
# 2008/06/20: CS0910309132
# 2012/03/20: CS1047210502
# 2013/06/18: CS1092910958
# 2014/05/22: CS1126311293
# 2017/02/19: CS1227012297
# 2017/03/29: CS1229812328
# 2017/05/16: CS1235912389
texts   = ['CS0523805267','CS0834108371','CS0867608705',
           'CS0910309132','CS1047210502','CS1092910958',
           'CS1126311293','CS1227012297',
           'CS1229812328','CS1235912389']

regions=['Squall line','Rear','Front']
#cloud_type_list = ['Ci', 'Cs','Dc','Ac', 'As','Ns','Cu', 'Sc','St']
cloud_type_list = ['Dc', 'Cs','Ci','NS', 'As','Ac','St', 'Sc','Cu']
ws_list = ['WS1', 'WS2','WS3','WS4', 'WS5','WS6','WS7', 'WS8','WS9','WS10','WS11']

#%% load global lat, lon data
lat = np.genfromtxt('lat.txt')
lon = np.genfromtxt('lon.txt')
lat_ws = np.arange(-90.5,89.5,1)
lon_ws = np.arange(0.5,360.5,1)
name = np.arange(1,11,1)  # 10 cases

# time for analysis
if num == 8 or num == 10:    
    i = 0
elif num == 3 or num == 7:
    i = 1
elif num == 6:
    i = 2
elif  num == 4 or num == 1 or num == 2 or num == 5 or num == 9:
    i = 3 
#%% set daomain edges
lat_s = [     21,    20,    14,    20,    23,    27,    15,    25,    22,     29]
lat_e = [     30,    32,    41,    50,    45,    44,    40,    40,    45,     45]
lon_s = [    117,   116,    98,   110,   245,   121,   100,   240,   250,    253]
lon_e = [    126, 128.5,   130,   140,   275,   136,   130,   263,   275,    268]

#%% set squall line region edges
slat_s = [  24.5,  22.5,    21,  27.5,    27,    30,    21,    27,    28,     33]
slat_e = [    29,  28.5,    31,    38,    36,  35.5,  26.5,    37,    35,   37.2]
slon_s = [ 119.5, 119.5,   111, 118.5,   263, 122.5,   107,   251,   263,  259,9]
slon_e = [   124,   125,   123,   130,   267,   135,   119,   256, 266.5,    265]

#%% functions
def mask_domain(num):
    lats = np.where((lat>=lat_s[num-1])&(lat<=lat_e[num-1]),lat,-1000)
    lons = np.where((lon>=lon_s[num-1])&(lon<=lon_e[num-1]),lon,-1000)
    mask_lat = np.equal(lats,-1000)
    mask_lon = np.equal(lons,-1000)

    return mask_lat, mask_lon

def mask_domain_ws(num):
    lats = np.where((lat_ws>=lat_s[num-1])&(lat_ws<=lat_e[num-1]),lat_ws,-1000)
    lons = np.where((lon_ws>=lon_s[num-1])&(lon_ws<=lon_e[num-1]),lon_ws,-1000)
    mask_lat_ws = np.equal(lats,-1000)
    mask_lon_ws = np.equal(lons,-1000)

    return mask_lat_ws, mask_lon_ws

def mask_cloud(cloud):
    mask_cloud = np.where(cloud==1,cloud,0)
    return mask_cloud

def connectpoints(ax,x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    ax.plot([x1,x2],[y1,y2],'k-',transform=ccrs.PlateCarree())
    
def connectpoints_1(ax,x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    ax.plot([x1,x2],[y1,y2],'r-',transform=ccrs.PlateCarree())
    
def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],pts[1][0]:pts[2][0]]
    return img_crop

def is_on_right_side(x, y, xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
    return a*x + b*y + c >= 0

def test_point(x, y, vertices):
    num_vert = len(vertices)
    is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
    all_left = not any(is_right)
    all_right = all(is_right)
    return all_left or all_right

def survey(fig, results, category_names):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('tab20b')(np.linspace(0.15, 0.85, data.shape[1]))

    #fig, ax = plt.subplots(figsize=(20, 3))
    ax = fig.add_subplot(3, 1, 3)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color, alpha=0.7)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        #ax.bar_label(rects, label_type='center', color=text_color)
    
    #ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
             #loc='upper left', fontsize='medium')
    ax.legend(ncol=len(category_names),
              loc='lower left', 
              fontsize='large',
              bbox_to_anchor=(-0.006, -0.2, 0.5, 0.5))

    return ax
