from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
### Import useful routines
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import *
from builtins import object
import numpy as np
import string
import glob
import os
from collections import Counter

### Import CDAT routines ###
import MV2 as MV
import cdms2 as cdms
import genutil
import cdutil
import cdtime
from eofs.cdms import Eof

### Import scipy routines for smoothing, interpolation
from scipy.interpolate import interp1d
from scipy.optimize import brentq,fminbound
import scipy.ndimage as ndimag

import CMIP5_tools as cmip5

### Import plotting routines
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import mpl

### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)

################################################################
#                                                              #
#                     Plotting routines                        #
#                                                              #
################################################################

def plotcdms(i,**kwargs):
    """ plot a cdms 1d variable """
    plt.plot(get_plottable_time(i),i.asma(),**kwargs)

def find_ij(data, xxx_todo_changeme):
    (xy) = xxx_todo_changeme
    lat = data.getLatitude()[:]
    lon = data.getLongitude()[:]
    x=xy[0]
    y=xy[1]
    i = np.argmin(np.abs(lat-x))
    j = np.argmin(np.abs(lon-y))
    return i,j


def bmap(X,projection="moll",**kwargs):
    """ quick plot of data on a lat,lon grid """
   # lon = X.getLongitude()[:]
    #lat = X.getLatitude()[:]
    
    lon = X.getLongitude().getBounds()[:,0]
    lat = X.getLatitude().getBounds()[:,0]
    
    if not ("lon_0" in list(kwargs.keys())):
        lon_0=np.median(lon)
    else:
        lon_0=kwargs.pop("lon_0")
        X,lon = shiftgrid(180,X,lon,start=False)
    m = Basemap(lon_0=lon_0,projection=projection,**kwargs)
    
        
    x,y=m(*np.meshgrid(lon,lat))
    #if vmin is None:
    m.pcolormesh(x,y,X,**kwargs)
    #else:
     #   m.pcolor(x,y,X,vmin=vmin,vmax=vmax)
    return m

class InteractiveMap(object):
    def __init__(self,data,proj="moll",typ="clim",fix_colorbar = True,**kwargs):
       # matplotlib.rcParams["backend"]="TkAgg"

        if data.id.find("pr")==0:
            lab = "mm/day/decade"
        else:
            lab = "K/decade"
        if len(data.shape) == 4:
            self.avg = MV.average(data,axis=0)
            self.data = data
        else:
            self.avg = data
            self.data = MV.array(data.asma()[np.newaxis])
            for i in range(3):
                self.data.setAxis(i+1,data.getAxis(i))
        
    
        if typ == "slopes":
            self.plotdata= genutil.statistics.linearregression(self.avg,nointercept=1)*3650.
        elif typ == "clim":
            self.plotdata = MV.average(self.avg,axis=0)
        elif typ == "eof":
            eofdata = cdms_clone(self.avg.anom(axis=0),self.avg)
            solver = Eof(eofdata)
            fac = get_orientation(solver)
            self.plotdata = solver.eofs()[0]*fac
        
        if fix_colorbar:
            a = max([np.abs(np.min(self.plotdata)),np.abs(np.max(self.plotdata))])
            vmin = -a
            vmax = a
        else:
            vmin=None
            vmax = None
       
        self.m = bmap(self.plotdata,alpha=1,projection=proj,vmin=vmin,vmax=vmax)
        self.m.drawcoastlines()
        plt.set_cmap(cm.RdBu_r)
        cbar=plt.colorbar(orientation="horizontal")
        cbar.set_label(lab)
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.lat = data.getLatitude()
        self.latbounds = data.getLatitude().getBounds()
        self.lon = data.getLongitude()
        self.cid = self.fig.canvas.mpl_connect('button_press_event',self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect("key_press_event",self.onpress)
        self.key = "o"
        self.stars = []
        self.figs=[]
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()
        

    def onpress(self,event):
        self.key = event.key
        #print self.key
        if self.key == "c":
            print("CLEARING ALL")
            [x.set_visible(False) for x in self.stars]
            [plt.close(fig) for fig in self.figs]
            self.fig.canvas.draw()
            self.key = "o" #reset
            return
        if self.key == "d":
            self.fig.canvas.mpl_disconnect(self.cid)
        if self.key == "z":
            print("ZOOM MODE ON")
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = self.fig.canvas.mpl_connect('button_press_event',self.zoom)
        if self.key == "o":
            print("ZOOM MODE OFF")
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = self.fig.canvas.mpl_connect('button_press_event',self.onclick)
        if self.key == "r":
            print("RESETTING")
            self.fig.canvas.mpl_disconnect(self.cid)
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.fig.canvas.draw()
            self.cid = self.fig.canvas.mpl_connect('button_press_event',self.onclick)
            #self.key = "o"
    def zoom(self,event):
        
        x,y = event.xdata, event.ydata
        lon,lat = self.m(x,y,inverse=True)
        xl,yl = self.m(lon-10,lat-10)
        xu,yu = self.m(lon+10,lat+10)
        self.ax.set_xlim(xl,xu)
        self.ax.set_ylim(yl,yu)
        self.fig.canvas.draw()
    def onclick(self,event):
        if not event.inaxes:
            return
        #if True:
        x = event.xdata
        y = event.ydata

        lon,lat = self.m(x,y,inverse=True)
        if self.m.lonmin > -100.:
            if lon < 0:
                lon = 360+lon


        xy = (lat,lon)
        t = get_plottable_time(self.data)
        i,j = find_ij(self.data,xy)



        X,Y = self.m(self.lon[j],self.lat[i])
        self.stars += [self.m.plot(X,Y,"y*",markersize=15)[0]]
        f2 = plt.figure()
        self.figs+= [f2]

        ax2 = f2.add_subplot(111)

        plt.draw()
        for mod in range(self.data.shape[0]):

            ax2.plot(t,self.data[mod,:,i,j].asma(),color=cm.gray(.5))
        ax2.plot(t,MV.average(self.data,axis=0)[:,i,j].asma(),color="k")
        ax2.set_title("("+str(self.lat[i])+","+str(self.lon[j])+")")
        plt.draw()
        
    def onselect(self, verts):
        path = Path(verts)
        #self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        #self.fc[:, -1] = self.alpha_other
        #self.fc[self.ind, -1] = 1
        #self.collection.set_facecolors(self.fc)
        self.fig.canvas.draw_idle()


#from mpl_toolkits.basemap import Basemap,shiftgrid
# def bmap(X,**kwargs):
#     """ quick plot of data on a lat,lon grid """
#     lon = X.getLongitude()[:]
#     lat = X.getLatitude()[:]
    
#     m = Basemap(lon_0=np.median(lon),projection="moll")
    
        
#     x,y=m(*np.meshgrid(lon,lat))
    
#     m.pcolor(x,y,X,**kwargs)
   
#     return m


# def bmap_rect(X,vmin=None,vmax=None,lon_0=None):
#     """ quick plot of data on a lat,lon grid """
#     lon = X.getLongitude()[:]
#     lat = X.getLatitude()[:]
#     if lon_0 == None:
#         lon_0=0
#     X,lon = shiftgrid(180,X,lon,start=False)
#     m = Basemap(X,lon_0=lon_0)
#     #m=Basemap()
    
        
#     x,y=m(*np.meshgrid(lon,lat))
#     if vmin is None:
#         m.pcolor(x,y,X)
#     else:
#         m.pcolor(x,y,X,vmin=vmin,vmax=vmax)
#     return m
def cmap(X,**kwargs):
    """ quick plot of data on a lat,lon grid """
    lon = X.getLongitude()[:]
    lat = X.getLatitude()[:]
    
    m = Basemap(lon_0=np.median(lon),projection="moll")
    
        
    x,y=m(*np.meshgrid(lon,lat))
    
    m.contourf(x,y,X)
    
    return m


def lat_plot(x,ax=None,**kwargs):
    """plot a cdms zonal average"""
    lat = x.getLatitude()[:]
    if ax is None:
        ax=plt.gca()
    ax.plot(lat,x.asma(),**kwargs)

def plot_all_lats(x,**kwargs):
    cmap = kwargs.pop("cmap",cm.RdYlBu)
    nt = len(x.getTime())
    for i in range(nt):
        lat = x.getLatitude()[:]
        plt.plot(lat,x[i].asma(),color=cmap(i/float(nt)))
        
def time_plot(x,ax=None,**kwargs):
    """ plot a cdms time series """
    t = cmip5.get_plottable_time(x)
    if ax is None:
        ax=plt.gca()
    
    ax.plot(t,x.asma(),**kwargs)
def label_theta_ticks(ax):
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    monthstarts = np.array([cdtime.comptime(0o001,x,1).torel("days since 0001-1-1").value for x in np.arange(12)+1])/365. *2*np.pi
    ax.set_xticks(monthstarts)
    ax.set_xticklabels(months)

def latitude_label_ticks(ax,axis='x'):
    latlabels=[]
    func = getattr(ax,"get_"+axis+"ticks")
    for lat in func():
        if lat <0:
            latlabels +=[str(int(-1*lat))+r'$^{\circ}$S']
        elif lat ==0:
            latlabels +=[str(int(lat))+r'$^{\circ}$']
        else:
            latlabels +=[str(int(lat))+r'$^{\circ}$N']
    func2=getattr(ax,"set_"+axis+"ticklabels")
    func2(latlabels)

def prep_for_talk(ax):
    ax.tick_params(colors="w")
    ax.xaxis.label.set_color("w")
    ax.yaxis.label.set_color("w")
    try:
        ax.legend_.get_frame().set_alpha(0.3)
    except:
        print("no legend")
    ax.title.set_color("w")

    plt.draw()


def model_dictionary():
    models = ['ACCESS1-0', 'ACCESS1-3', 'BNU-ESM','CCSM4', 'CESM1-BGC',\
       'CESM1-CAM5', 'CESM1-FASTCHEM', 'CESM1-WACCM', 'CMCC-CESM',\
       'CMCC-CM','CMCC-CMS', 'CNRM-CM5', 'CNRM-CM5','CNRM-CM5-2', 'CSIRO-Mk3-6-0', 'CanESM2',\
       'FGOALS-g2', 'FGOALS-s2', 'FIO-ESM', 'GFDL-CM3', 'GFDL-ESM2G',\
       'GFDL-ESM2M', 'GFDL-HIRAM-C180','GFDL-HIRAM-C360','GISS-E2-H*p1', 'GISS-E2-H*p3', 'GISS-E2-H-CC',\
       'GISS-E2-R*p1', 'GISS-E2-R*p3', 'GISS-E2-R-CC', 'HadCM3',\
       'HadGEM2-AO', 'HadGEM2-CC', 'HadGEM2-ES', 'IPSL-CM5A-LR',\
       'IPSL-CM5A-MR', 'IPSL-CM5B-LR', 'MIROC-ESM', 'MIROC-ESM-CHEM',\
       'MIROC4h', 'MIROC5', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MPI-ESM-P','MRI-CGCM3',\
       'NorESM1-M', 'NorESM1-ME', 'bcc-csm1-1', 'bcc-csm1-1-m', 'fio-esm',\
       'inmcm4']
    markers =["o","v","^","<",">","8","s","p","*","h","H","D","d"]#,"P","X"]
    Lm = len(markers)
    d={}
    i=0
    colors = [cm.Set1(i/9.) for i in range(9)]+[cm.Set2(i/9.) for i in range(9)]+[cm.Set3(i/9.) for i in range(9)]
    Lc = len(colors)
    for i in range(len(models)):
        model =models[i]
        d[model]= {}
        d[model]["color"]=colors[np.mod(i,Lc)]
        d[model]["marker"]=markers[np.mod(i,Lm)]
    d["Can*"]=d["CanESM2"]
    d["CanAM4"]=d["CanESM2"]
    d["HadGEM2-A*"]=d["HadGEM2-AO"]
    d["HadGEM2-A"]=d["HadGEM2-AO"]
    return d
        
def scatterplot_cmip(X,Y):
    """
    Scatterplot the arrays X and Y.  If ensemble_average, just plot.  Otherwise, group ensemble members by model.  X and Y must be of the same length and have the same model axes.
    """
    if len(cmip5.models(X)[0].split("."))==1:
        ensemble_average=True
    else:
        ensemble_average=False
    markers = model_dictionary()
    if not ensemble_average:
        ed = cmip5.ensemble_dictionary(X)
        models = sorted(ed.keys())
        for i in range(len(models)):
            
            model = models[i]
            print(model)
            c = markers[model]["color"]
            marker=markers[model]["marker"]
            plt.plot(X.asma()[ed[models[i]]],Y.asma()[ed[models[i]]], marker,markersize=10,color=c,label=models[i])
    else:
        models = cmip5.models(X)
        for i in range(len(models)):
            model = models[i]
            c = markers[model]["color"]
            marker=markers[model]["marker"]
            plt.plot([X[i]],[Y[i]], marker,markersize=10,color=c,label=models[i])
