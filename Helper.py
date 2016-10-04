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


#########################################
#                                       #
#        Useful helper functions        #  
#                                       #
#########################################

def check_for_continuity(fname,interval = 'monthly'):
    """ Checks a netcdf or xml file for continuity in the time axis"""
    f = cdms.open(fname)
    tax = f.getAxis('time')
    spacing = np.diff(tax[:])
    unit = tax.units.split(" ")[0]
    if unit == "days":
        if interval == 'monthly':
            maxspacing = 31.
        elif interval == "yearly":
            maxspacing = 365.
    elif unit == "months":
        if interval == 'monthly':
            maxspacing == 12.
    elif unit == "years":
        maxspacing = 1.
    toobig = np.where(spacing > maxspacing)[0]
    f.close()
    
    if len(toobig) >0:
        return False
    else:
        return True
def get_ensemble(basedir,model,search_string = "*"):
    if model.find("GISS")>=0:
        if search_string == "*":
            search_string = "*p1.*"
        else:
            
            
            model = model.split(" ")[0]
        
        alldata = glob.glob(basedir+"*"+model+"."+search_string)
        
    else:
        alldata = glob.glob(basedir+"*."+model+"."+search_string)
    modified = []
    truncated = np.array(map(lambda x: x.split("ver")[0], alldata))
    c = Counter(truncated)
    for k in c.keys():
        I = np.where(truncated==k)[0]
        listoffiles = np.array(alldata)[I]
        if c[k]>1:
            modified += [get_latest_version(listoffiles)]
        else:
            modified += [listoffiles[0]]
    return modified



def get_latest_version(listoffiles):
    version_numbers = map(lambda x: x.split("ver-")[-1].split(".")[0], listoffiles)
    enums = []
    for num in version_numbers:
        
        if num[0] != "v":
            enums+=[int(num)]
        else:
            enums+=[int(num[1:])]
    
    i = np.argmax(enums)
    return listoffiles[i]

def spatially_smooth(data,sigma=5):
    """Spatially smooth even when spatial data is missing """
    # Convert resolution to grid cells from latitude
    resolution = np.median(np.diff(data.getLatitude()[:]))
    nt = data.shape[0]
    sigma = sigma/resolution
    D = MV.zeros(data.shape)+1.e20
    
    #If there is no missing data, we can just smooth
    if not hasattr(data,"mask"):
        data.mask=False
    if data.mask.shape!= data.shape:        
        for i in range(nt):
            D[i] = ndimag.gaussian_filter(data[i],sigma=sigma)
    #Otherwise, apply smoothing filter to compressed data
    else:
        for i in range(nt):
            datai = data[i]

            if len(datai.compressed())!=0: #if not all the data is missing at that time step
                datai = MV.array(ndimag.gaussian_filter(datai.compressed(),sigma))
                datai = MV.masked_where(datai>1.e10,datai)
                J = np.where(~data[i].mask)[0]
                
                D[i,J] = datai
        # For data where all lats are masked (ie, missing time step)
        D = MV.masked_where(D>1.e10,D)

    D.setAxis(0,data.getTime())
    D.setAxis(1,data.getLatitude())
    return D


################################################################
#                                                              #
#                     File management functions                #
#                                                              #
################################################################
def get_plottable_time(X):
    years = [x.year+(x.month-1)/12. for x in X.getTime().asComponentTime()]
    return np.array(years)

def cdms_clone(X,Y):
    X = MV.array(X)
    X.setAxisList(Y.getAxisList())
    for key in Y.attributes.keys():
        setattr(X,key,Y.attributes[key])
    X.id = Y.id
    return X

def check_length(cl_fname):
    """Quickly get the length of the time dimension"""
    f=cdms.open(cl_fname)

    t=len(f.getAxis('time'))
    f.close()
    return t

def get_too_short(fnames,bound):
    """ Return the indices of filenames in an array with time axis length shorter than bound """
    too_short = np.where([check_length(x)<bound-1 for x in fnames])[0]
    
    return too_short


def get_common_timeax(fnames):
    """Find the end point (non-historical) or start/stop times (historical) such that all fimes in [fnames] have the same time axis.  Return an integer (non-historical) or start-stop times (historical)."""
    path = fnames[0].split("cmip5.")[0]
    if path.find("historical")<0:
        L = []
        #badrips = 
        for fname in fnames:
            f = cdms.open(fname)
            L += [len(f.getAxis('time'))]
            if len(f.getAxis('time'))<50:
                print fname
                print len(f.getAxis('time'))
            f.close()
            #Maybe this is a stupid assumption, but take all models as starting at the same time.  CHECK THIS
        if len(np.unique(L))>1:
            truncate=np.min(L)
        else:
            truncate = None
        return truncate
    else:
        
        
        #get starts and stops
        starts = []
        stops = []
        
        for model in fnames:
            f = cdms.open(model)
            tax = f.getAxis("time").asComponentTime()
#            if tax[-1].year < 2050:
 #               print model
  #              print tax[-1]
            starts+=[tax[0]]
            stops+=[tax[-1]]
            
            
            f.close()

        return max(starts),min(stops)
def get_orientation(solver):
    pc1 = solver.pcs()[:,0]
    fac = float(np.sign(genutil.statistics.linearregression(pc1,nointercept=1)))
    return fac


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
        


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
     
    raise TypeError('shape must be an int, or a tuple of ints')
 
from numpy.lib.stride_tricks import as_strided as ast
 
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
     
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.
     
    Returns
        an array containing each n-dimensional window from a
    '''
     
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
     
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
     
     
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
     
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
     
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
     
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


def giss_forcing_lookup(forcing,variable,model="R",p=1):
    """ get GISS forcings"""
    forcing_search = "*p"+str(p)
    if forcing.find("Nat")>=0:
        path = "/work/cmip5/historicalNat/atm/mo/"+variable+"/"
        
    elif forcing.find("hist")>=0:
        path = "/work/cmip5/historical/atm/mo/"+variable+"/"    
    elif forcing.find("GHG")>=0:
        if forcing =="LLGHG":
            path = "/work/cmip5/historicalMisc/atm/mo/"+variable+"/"
        else:
            path = "/work/cmip5/historicalGHG/atm/mo/"+variable+"/" 
    else:
      path = "/work/cmip5/historicalMisc/atm/mo/"+variable+"/"

   
    d = {}
    d["Sl"]= '02'
    d["Vl"]= '03'
    d["LU"]= '04'
    d["Oz"]= '14'
    d["Oz_105"]='05'
    
    d["AA"] = ['310','107']
    d["CH4"] = '11'
    d["CFCs"] = '12'
    d['CO2'] = '13'
    d["Ant"] = '09'
    d["LLGHG"]= '13'
    
    
    model_search = "GISS-E2-"+model
    
    if forcing in d.keys():
        
        if forcing != "AA":
            forcing_search = forcing_search+d[forcing]
            
            
        else:
            test = d["AA"]
            ok=test[int(np.where([x[0] == str(p) for x in test])[0])]
            forcing_search = "*p"+str(ok)
    else:
       
        if path.find("historicalMisc")>=0:
            return []
    
    forcing_search = forcing_search+".*"
    
   # print path
   # print model_search
   # print forcing_search
    return sorted(get_ensemble(path,model_search,forcing_search))

def HistoricalMiscLookup(forcing,variable):
    d = {}
    d["Ant"] = ["*GFDL-ESM2M*.r1i1p2.*",\
           "*CSIRO*.r*i1p1.*" ,\
           "*GFDL-CM3*.r[135]i1p2.*",\
           "*GISS*.r*i1p109.*",\
           "*GISS*.r*i1p309.*",\
           "*CNRM-CM5*.r*i1p1.*",\
           "*IPSL-CM5-LR*.r[123]i1p2.*",\
           "*CCSM4*.r[1,4,6]i1p11.*",\
           "*CESM1-CAM5*.r[1,4,6]i1p11.*"]
    
    d["AA"] = ["*GFDL-ESM2M*.r1i1p5.*",\
          "*CanESM2*.r[1-5]i1p4.*",\
          "*CSIR0*.r*i1p4.*",\
          "*GISS*.r*i1p107.*",\
          "*NorESM1-M*.r1i1p1.*",\
          "*FGOALS*.r2i1p1.*",\
          "*IPSL-CM5-LR.r1i1p3.*",\
          "*GISS*.r[1-5]i1p310.*",\
          "*GFDL*.r[135]i1p1.*",\
          "*CCSM4*.r[1,4,6]i1p10.*",\
          "*CESM1-CAM5*.r[1,4,6]i1p10.*"]
        
    #### Single forcing experiments ######            
    
    d["Vl"] = ["*GFDL-ESM2M*.r1i1p8.*",\
          "*GISS-E2*.r[1-5]i1p[13]03.*",\
          "*CSIRO*r*i1p6*",\
          "*CCSM4*.r[1,4,6]i1p17.*",\
          "*CESM1-CAM5*.r[1,4,6]i1p17.*"]
          
    d["Oz"] = ["*GISS*i1p105*",\
    "*FGOALS*r1i1p1*",\
    "*CCSM4*.r[1,4,6]i1p14.*",\
    "*CESM1-CAM5*.r[1,4,6]0i1p14.*"]

    
    

    d["Sl"] = ["*GFDL-ESM2M*.r1i1p7*",\
          "*CanESM2*r[1-5]i1p3.*",\
          "*GISS*r[1-5]i1p[13]02*",\
          "*CCSM4*.r[1,4,6]i1p16.*",\
          "*CESM1-CAM5*.r[1,4,6]i1p17.*"]
    
    
    
    d["LU"] = ["*GFDL-ESM2M*r1i1p6.*",\
          "*CanESM2*.r[1-5]i1p2.*",\
          "*GISS*.r[1-5]i1p104.*",\
          "*CCSM4*.r[1,4,6]i1p13.*",\
          "*CESM1-CAM5*.r[1,4,6]i1p13.*"]

    if forcing.find("Nat")>=0:
        path = "/work/cmip5/historicalNat/atm/mo/"+variable+"/"
        
    elif forcing.find("hist")>=0:
        path = "/work/cmip5/historical/atm/mo/"+variable+"/"    
    elif forcing.find("GHG")>=0:
        if forcing =="LLGHG":
            path = "/work/cmip5/historicalMisc/atm/mo/"+variable+"/"
        else:
            path = "/work/cmip5/historicalGHG/atm/mo/"+variable+"/" 
    elif forcing in d.keys():
      path = "/work/cmip5/historicalMisc/atm/mo/"+variable+"/"
    else:
        print "forcing must be one of "+str(d.keys()+["GHG","Nat","hist","LLGHG"])
        raise TypeError

    if forcing in d.keys():
        search_strings = d[forcing]
    else:
        search_strings = ["*"]
    selected = []
    for ss in search_strings:
        candidates = glob.glob(path+ss)
            
        stems = np.unique([x.split("ver")[0] for x in candidates])
        for stem in stems:
            selected += [get_latest_version(glob.glob(stem+"*"))]
    return selected
                
            

########### Stuff from the ECS paper #######

def get_tas(forcing,average=True):
    """ Read in pre-computed annual-average, global-average surface air temperatures (NOT anomalies)"""
    if forcing == 'GHG':
        forcing = 'historicalGHG'
    model = "GISS-E2-R"
    p = "*p1*"
    direc = "/Users/kmarvel/Google Drive/HistoricalMisc/GLOBAL_MEAN/"+forcing+"/"
    files = get_ensemble(direc,model,search_string = "*p1*YEAR*")
    f = cdms.open(files[0])
    c = 0
    f = cdms.open(files[0])
    start = f["tas"].getTime().asComponentTime()[0]
    stop = f["tas"].getTime().asComponentTime()[-1]
    tax = f("tas").getTime()
    L = len(f("tas",time=(start,stop)))
    tas = MV.zeros((len(files),L))
    f.close()
    for fil in files:
        f = cdms.open(fil)
        tas[c] = f("tas",time=(start,stop))
       
        c+=1
    if average:
        temperatures = MV.average(tas,axis=0)
   
        temperatures.setAxis(0,tax)
    else:
        temperatures = tas
        temperatures.setAxis(1,tax)
    return temperatures
def get_ohc(forcing,average=True,prefix = "/Users/kmarvel/Google Drive/ECS/OHC_DATA/"):
    """ read in the ocean heat content"""
    direc = prefix+forcing+"/"
    files = sorted(glob.glob(direc+"*.nc"))
    L = len(files)
    f = cdms.open(files[0])
    ohc_raw = f("ohc")
    tax = ohc_raw.getTime()
    ohc_sum = MV.sum(ohc_raw, axis=1)
    ohc_final = (ohc_sum)*1.e-22 #Convert to 1e22 Joules
    nt = len(ohc_final)
    MMA = MV.zeros((L,nt))
    MMA[0]=ohc_final
    f.close()
    
    #Loop over all files in directory (ensemble members)
    for i in range(L)[1:]:
        f = cdms.open(files[i])
        ohc_raw = f("ohc")
        ohc_sum = MV.sum(ohc_raw, axis=1)
        ohc_final = (ohc_sum)*1.e-22
        MMA[i]=ohc_final
        f.close()
        
    #Return multilodel average, or not
    if average:
        MMA = MV.average(MMA,axis=0)
        MMA.setAxis(0,tax)
    else:
        MMA.setAxis(1,tax)
    return MMA
