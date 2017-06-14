### Import the usual suspects
import glob
import sys
import os
import numpy as np
import string


### Import things from cdutil
import MV2 as MV
import cdms2 as cdms
import cdtime,cdutil,genutil
#from eofs.cdms import Eof
#from eofs.multivariate.cdms import MultivariateEof

### plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm


### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)


global crunchy
import socket
if socket.gethostname().find("crunchy")>=0:
    crunchy = True
else:
    crunchy = False





################################################################
#                                                              #
#                     File management functions                #
#                                                              #
################################################################



def get_latest_version(listoffiles):
    """ on crunchy: find the latest version of a list of files. Returns single file. """
    version_numbers = map(lambda x: x.split("ver-")[-1].split(".")[0], listoffiles)
    enums = []
    for num in version_numbers:
        
        if num[0] != "v":
            enums+=[int(num)]
        else:
            enums+=[int(num[1:])]
    
    i = np.argmax(enums)
    return listoffiles[i]


def only_most_recent(allfiles_nover):
    #version control: take only the newest version of each file in a list of files (on crunchy)
    allfiles = []
    uniq=np.unique([x.split(".ver")[0] for x in allfiles_nover])
    for stem in uniq:
        fnames = glob.glob(stem+"*")
        allfiles+=[get_latest_version(fnames)]
    return allfiles



def cdms_clone(X,Y):
    """X is a numpy array, Y is a MV array.  Transform X into an MV array and give it all the attributes of Y"""
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
    """ orient an Eof instance so that the trend in PC1 is positive """
    pc1 = solver.pcs()[:,0]
    fac = float(np.sign(genutil.statistics.linearregression(pc1,nointercept=1)))
    return fac

def get_nearest_latitude_index(Z,lat):
    lats = Z.getLatitude()[:]
    return np.argmin(np.abs(lats-lat))

def get_nearest_latitude(Z,lat):
    lats = Z.getLatitude()[:]
    return Z[np.argmin(np.abs(lats-lat))]

class HistoricalMisc():
    """ Single-forcing simulations """
    def __init__(self):
        ###### Special Collections of forcings ######        

        self.Ant = ["*GISS-E2-R.*r*i1p109.*",\
                    "*GISS-E2-H.*r*i1p109.*",\
                    "*GISS-E2-R.*r*i1p309.*",\
                    "*GISS-E2-H.*r*i1p309.*",\
                    "*CCSM4*.r[1,4,6]i1p11.*",\
                    "*GFDL-ESM2M.*r1i1p2.*",\
                    "*CSIRO-Mk3-6-0*r[1-10]i1p1.*",\
                    "*GFDL-CM3.*r[1,3,5]i1p2.*",
                    "*CNRM-CM5.*r[1-10]i1p1.*",\
                    "*IPSL-CM5A-LR.*r[1-3]i1p2.*"]

        self.AA = ["*GISS-E2-R.*r[1-5]i1p310.*","*GISS-E2-H.*r[1-5]i1p310.*","*GISS-E2-H.*r[1-5]i1p107.*","*GISS-E2-R.*r[1-5]i1p107.*","*CCSM4*.r[1,4,6]i1p10.*","*GFDL-ESM2M.*r1i1p5.*","*CanESM2*r[1-5]i1p4.*","*CSIRO-Mk3-6-0*r[1-10]i1p4.*","*NorESM1-M.*r1i1p1.*", "*IPSL-CM5-LR.*r1i1p3.*","*GFDL-CM3.*r[1,3,5]i1p1.*" ]


        #### Single forcing experiments ######            

        self.Vl = [ "*GISS-E2-R.*r[1-5]i1p103.*","*GISS-E2-H.*r*[1-5]i1p103.*","*CCSM4*.r[1,4,6]i1p17.*","*GFDL-ESM2M.*r1i1p8.*","*CSIRO-Mk3-6-0*r[1-10]i1p6.*"]

        self.Sl = [  "*GISS-E2-H.*r[1-5]i1p[13]02*","*GISS-E2-R.*r[1-5]i1p[13]02*","*CCSM4*.r[1,4,6]i1p16.*", "*GFDL-ESM2M.*r1i1p7.*", "*CanESM2*r[1-5]i1p3.*"]

        self.Oz = ["*GISS-E2-R.*r[1-5]i1p114*","*GISS-E2-H.*r[1-5]i1p114*","*CCSM4*.r[1,4,6]i1p14.*"]

        self.LU = [ "*GISS-E2-R.*r[1-5]i1p104.*", "*GISS-E2-H.*r[1-5]i1p104.*","*CCSM4*.r[1,4,6]i1p13.*", "*GFDL-ESM2M.*r1i1p6.*", "*CanESM2*r[1-5]i1p2.*"]

def get_datafiles(forcing,variable,realm="atm"):
    """ List all files on crunchy corresponding to a variable"""
    if not crunchy:
        print "Not on crunchy"
        raise TypeError()
    hm = HistoricalMisc()
    if hasattr(hm,forcing):
        experiment = "historicalMisc"
        search_strings = getattr(hm,forcing)
    elif forcing == "abrupt4xCO2":
        experiment = forcing
        search_strings = ["*r1*"]
    else:
        experiment = forcing
        search_strings = ["*"]
    direc = "/work/cmip5/"+experiment+"/"+realm+"/mo/"+variable+"/"
    fnames = []
    for search_string in search_strings:
        candidates = glob.glob(direc+search_string)
        for stem in np.unique([x.split("ver")[0] for x in candidates]):
            fnames += [get_latest_version(glob.glob(stem+"*"))]
    return sorted(fnames)


def boring(x):
    return x   
def multimodel_average(direc,variable,*args,**kwargs):
    """multimodel average over all files in directory that match search string (default *).  Apply func to data (default identity)"""
    #default values:
    search_string = kwargs.pop("search_string","*")
    func = kwargs.pop("func",boring)
    verbose = kwargs.pop("verbose",False)
   
    #All files in the directory that match the criteria
    allfiles = np.array(glob.glob(direc+search_string))
    models = np.unique([x.split(".")[1] for x in allfiles])
    ensemble_dictionary={}
    for model in sorted(models):
         #separate GISS-E2-R p1 and p3
        if model == "GISS-E2-R":
            giss_r = allfiles[np.where(np.array([x.find("GISS-E2-R.")>=0 for x in allfiles]))]
            ensemble_dictionary["GISS-E2-R*p3"] = giss_r[np.where(np.array([x.find("p3")>=0 for x in giss_r]))]
            ensemble_dictionary["GISS-E2-R*p1"] = giss_r[np.where(np.array([x.find("p3")>=0 for x in giss_r]))]
        elif model == "GISS-E2-H":
             #separate GISS-E2-H p1 and p3
            giss_h = allfiles[np.where(np.array([x.find("GISS-E2-H.")>=0 for x in allfiles]))]
            ensemble_dictionary["GISS-E2-H*p3"] = giss_h[np.where(np.array([x.find("p3")>=0 for x in giss_h]))]
            ensemble_dictionary["GISS-E2-H*p1"] = giss_h[np.where(np.array([x.find("p3")>=0 for x in giss_h]))]
        else:
            ensemble_dictionary[model]=allfiles[np.where(np.array([x.find("."+model+".")>=0 for x in allfiles]))]
            
            
            
    
    #Now make a MMA to hold all model esembles
    
    effective_models = sorted(ensemble_dictionary.keys())
    nmod = len(effective_models)
    #get the shape of the thing we're averaging
    try:
        testfile = sorted(ensemble_dictionary["CCSM4"])[0] #Assume CCSM4 r1 will have the representative shape
    except:
        testfile = sorted(ensemble_dictionary["MIROC5"])[0] #try miroc
    
    f = cdms.open(testfile)
    test = func(f(variable),*args,**kwargs)
    data_shape = test.shape
    data_axes = test.getAxisList()
    
    MMA = MV.zeros((nmod,)+data_shape)+1.e20
    if verbose:
        print "MMA shape will be "+str(MMA.shape)
    f.close()
    #Now average over each ensemble
    for i in range(nmod):
        key = effective_models[i]
        ensemble = ensemble_dictionary[key]
        nens = len(ensemble)
        ENS = MV.zeros((nens,)+data_shape)+1.e20
        for j in range(nens):
            try:
                f = cdms.open(ensemble[j])
                if verbose:
                    print ensemble[i]
                ENS[j]=func(f(variable),*args,**kwargs)
                f.close()
            except:
                print ensemble[j] +" has a problem"
        #ENS = MV.masked_where(ENS>1.e10,ENS)
        MMA[i] = MV.average(ENS,axis=0)
    MMA = MV.masked_where(MMA>1.e10,MMA)
    modax = cdms.createAxis(np.arange(nmod).astype(np.float))
    modax.models = str(effective_models)
    modax.id = "model"
    modax.name = "model"
    modax.long_name = "CMIP5 model"
    
    MMA.setAxisList([modax]+data_axes)
    return MMA
    
    
    
def get_ensemble(direc,variable,*args,**kwargs):
    """get all files in directory that match search string (default *).  Apply func to data (default identity)"""
    #default values:
    search_string = kwargs.pop("search_string","*")
    func = kwargs.pop("func",boring)
   
    #All files in the directory that match the criteria
    allfiles_nover = np.array(sorted(glob.glob(direc+search_string)))

    #version control: take only the newest 
    allfiles =  only_most_recent(allfiles_nover)
   
    nfiles = len(allfiles)

    #Get the shape 
    f = cdms.open(allfiles[0])
    test = func(f(variable),*args,**kwargs)
    data_shape = test.shape
    data_axes = test.getAxisList()
    
    ENSEMBLE = MV.zeros((nfiles,)+data_shape)+1.e20
    f.close()
    #Now read in every member of the ensemble
    for i in range(nfiles):
        f=cdms.open(allfiles[i])
        try:
            ENSEMBLE[i] = func(f(variable),*args,**kwargs)
            f.close()
        except:
            f.close()
            continue
    ENSEMBLE= MV.masked_where(ENSEMBLE>1.e10,ENSEMBLE)
    modax = cdms.createAxis(np.arange(nfiles).astype(np.float))
    modax.models = str(allfiles)
    modax.id = "filename"
    modax.name = "filename"
    modax.long_name = "File name"
    
    ENSEMBLE.setAxisList([modax]+data_axes)
    return ENSEMBLE
    
def start_time(data):
    return data.getTime().asComponentTime()[0]
def stop_time(data):
    return data.getTime().asComponentTime()[-1]
def get_plottable_time(X):
    years = [x.year+(x.month-1)/12. for x in X.getTime().asComponentTime()]
    return np.array(years)
def time_anomalies(data,start=None,stop=None):
    taxis = data.getAxisIds().index('time')
    if start is None:
        start = start_time(data)
    if stop is None:
        stop = stop_time(data)
    clim = MV.average(data(time=(start,stop)),axis=taxis)
    clim_exp  = np.ma.expand_dims(clim.asma(),taxis)
    anom=cdms_clone(data.asma()-clim_exp,data)
    return anom
    
    

def clim_sens(model,verbose=False):
    """ get the climate sensitivity"""
    curdir=__file__.split("CMIP5_tools.py")[0]
    cs = open(curdir+"clim_sens.txt")
    lns = cs.readlines()
    cs.close()
    models = np.array([string.lower(x.split("\t")[0]).replace("_","-") for x in lns[2:]])
    
    sens = [x.split("\t")[2].split("\n")[0] for x in lns[2:]]
    
    i = np.argwhere(models == string.lower(model))
    if len(i)==1:
        i=i[0]
        if sens[i]!="":
            if verbose:
                print "ECS for "+model+" is "+str( float(sens[i])/2.)
            
            return float(sens[i])/2.
        else:
            if verbose:
                print "No ECS found for "+model
            return np.nan
    else:
        if verbose:
            print "More than one match for "+model
            print "Choose one of "+str(models[i])
        return np.nan


def make_model_axis(listoffiles,just_modelnames = False):
    ax = cdms.createAxis(range(len(listoffiles)))
    ax.id = "model"
    ax.name="model"
    ax.long_name = "Model name"
    if just_modelnames:
        models = np.unique([x.split(".")[1] for x in listoffiles]).tolist()
    else:
        models = listoffiles
    ax.models = str(models)
    return ax



def all_clim_sens():
    """ get the climate sensitivity"""
    curdir=__file__.split("CMIP5_tools.py")[0]
    cs = open(curdir+"clim_sens.txt")
    lns = cs.readlines()
    cs.close()
    models = np.array([string.lower(x.split("\t")[0]).replace("_","-") for x in lns[2:]])
    
    sens = [x.split("\t")[2].split("\n")[0] for x in lns[2:]]
    thesens = []
    for ecs in sens:
        if ecs != "":
            #print ecs
            thesens += [float(ecs)/2.]
            
    return thesens


def landfrac(fname):
    model = fname.split(".")[1]
    experiment = fname.split(".")[2]
    if experiment == "amip":
        if model == "CanAM4":
            model = 'CanESM2'
    land_direc = "/work/cmip5/fx/fx/sftlf/"
    candidates = glob.glob(land_direc+"*"+model+".*"+experiment+".*")
    if len(candidates)>0:
        return get_latest_version(candidates)
    else:
       
        candidates = glob.glob(land_direc+"*"+model+".*")
        if len(candidates)>0:
            return get_latest_version(candidates)
        else:
            raise TypeError("Can't find matching landfrac")
        
        
