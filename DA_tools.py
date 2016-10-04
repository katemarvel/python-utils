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

### Import plotting routines
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap  
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.mlab as mlab

### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)

def concatenate_this(piC):
    nmodc,ntc,nlatc=piC.shape
    piC_concatenate = piC.reshape((nmodc*ntc,nlatc))
    tax = cdms.createAxis(np.arange(0,nmodc*ntc*365,365)+15.5)
    tax.units = 'days since 0001-1-1'
    tax.id = "time"
    tax.designateTime()
    piC_concatenate.setAxis(0,tax)
    return piC_concatenate

def get_slopes(c,yrs,plot = False):
    """
    get non-overlapping trends in concatenated control run
    """
    trends = np.ma.array([])
    start = 0
    end = yrs
    if plot:
        
        Ntot = float(len(c)/yrs)
    while end < len(c):
        ctrunc = c[start:end]
        #if len(ctrunc.compressed()) == len(ctrunc):
        if True:
        # Express in "per decade" units.  All control runs are in days.
            slope0,intercept0 = genutil.statistics.linearregression(ctrunc)
            slope = slope0*3650.
            if plot:
                xax = get_plottable_time(ctrunc)
                plt.plot(xax,ctrunc.asma(),color = cm.RdYlBu(float(start/yrs)/Ntot))
                plt.plot(xax,float(slope0)*ctrunc.getTime()[:]+float(intercept0),color ="k",linewidth = 3)
            trends = np.append(trends,slope)
        start = end
        end += yrs
       
    trends = MV.masked_where(np.isnan(trends),trends)
    trends = trends.compressed()
    return trends


def get_orientation(solver):
    pc1 = solver.pcs()[:,0]
    fac = float(np.sign(genutil.statistics.linearregression(pc1,nointercept=1)))
    return fac

def fit_normals_to_data(C,**kwargs):
    a = kwargs.pop("a",None)
    if a is None:
        a = np.max(C)
        a = a + 0.5*a
    delta = a/25.
    xc = np.arange(-a,a+delta,delta)

    muc = np.ma.average(C)
    sigc = np.ma.std(C)
    fac = 1./sigc
    pdfc = mlab.normpdf(xc,muc,sigc)
    plt.plot(xc,pdfc,**kwargs)
