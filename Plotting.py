### Import useful routines
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

from CMIP5_tools import *

### Import plotting routines
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap  
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

def find_ij(data,(xy)):
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
    m = Basemap(lon_0=np.median(lon),projection=projection)
    
        
    x,y=m(*np.meshgrid(lon,lat))
    #if vmin is None:
    m.pcolormesh(x,y,X,**kwargs)
    #else:
     #   m.pcolor(x,y,X,vmin=vmin,vmax=vmax)
    return m

class InteractiveMap():
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
            print "CLEARING ALL"
            [x.set_visible(False) for x in self.stars]
            [plt.close(fig) for fig in self.figs]
            self.fig.canvas.draw()
            self.key = "o" #reset
            return
        if self.key == "d":
            self.fig.canvas.mpl_disconnect(self.cid)
        if self.key == "z":
            print "ZOOM MODE ON"
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = self.fig.canvas.mpl_connect('button_press_event',self.zoom)
        if self.key == "o":
            print "ZOOM MODE OFF"
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = self.fig.canvas.mpl_connect('button_press_event',self.onclick)
        if self.key == "r":
            print "RESETTING"
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


from mpl_toolkits.basemap import Basemap,shiftgrid
def bmap(X,vmin=None,vmax=None):
    """ quick plot of data on a lat,lon grid """
    lon = X.getLongitude()[:]
    lat = X.getLatitude()[:]
    
    m = Basemap(lon_0=np.median(lon),projection="moll")
    
        
    x,y=m(*np.meshgrid(lon,lat))
    if vmin is None:
        m.pcolor(x,y,X)
    else:
        m.pcolor(x,y,X,vmin=vmin,vmax=vmax)
    return m


def bmap_rect(X,vmin=None,vmax=None,lon_0=None):
    """ quick plot of data on a lat,lon grid """
    lon = X.getLongitude()[:]
    lat = X.getLatitude()[:]
    if lon_0 == None:
        lon_0=0
    X,lon = shiftgrid(180,X,lon,start=False)
    m = Basemap(X,lon_0=lon_0)
    #m=Basemap()
    
        
    x,y=m(*np.meshgrid(lon,lat))
    if vmin is None:
        m.pcolor(x,y,X)
    else:
        m.pcolor(x,y,X,vmin=vmin,vmax=vmax)
    return m
def cmap(X,**kwargs):
    """ quick plot of data on a lat,lon grid """
    lon = X.getLongitude()[:]
    lat = X.getLatitude()[:]
    
    m = Basemap(lon_0=np.median(lon),projection="moll")
    
        
    x,y=m(*np.meshgrid(lon,lat))
    
    m.contourf(x,y,X)
    
    return m


def lat_plot(x,**kwargs):
    """plot a cdms zonal average"""
    lat = x.getLatitude()[:]
    plt.plot(lat,x.asma(),**kwargs)
def time_plot(x,**kwargs):
    """ plot a cdms time series """
    t = get_plottable_time(x)
    plt.plot(t,x.asma(),**kwargs)
def label_theta_ticks(ax):
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    monthstarts = np.array([cdtime.comptime(0001,x,1).torel("days since 0001-1-1").value for x in np.arange(12)+1])/365. *2*np.pi
    ax.set_xticks(monthstarts)
    ax.set_xticklabels(months)

def latitude_label_ticks(ax):
    latlabels=[]
    for lat in ax.get_xticks():
        if lat <0:
            latlabels +=[str(int(-1*lat))+r'$^{\circ}$S']
        elif lat ==0:
            latlabels +=[str(int(lat))+r'$^{\circ}$']
        else:
            latlabels +=[str(int(lat))+r'$^{\circ}$N']
    ax.set_xticklabels(latlabels)
    
