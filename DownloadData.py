#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cdms2 as cdms
import MV2 as MV
import cdtime,cdutil,genutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import string
import glob
import scipy.stats as stats
# Local solution
# If running remotely, uncomment the following code:
# %%bash
# git clone https://github.com/katemarvel/CMIP5_tools
# import CMIP5_tools as cmip5
import sys,os
#sys.path.append("/Users/kmarvel/Google Drive/python-utils")
sys.path.append("../python-utils")
import CMIP5_tools as cmip5
import DA_tools
import Plotting

from eofs.cdms import Eof
from eofs.multivariate.cdms import MultivariateEof
get_ipython().run_line_magic('matplotlib', 'inline')

import requests
import pandas as pd

### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)

#external_drive='/Volumes/CMIP6/'
external_drive="/home/kdm2144/"


# In[2]:


NCA4regions={}
#Northwest (NW): (125°W–111°W, 42°N–49°N)
NCA4regions["NW"]=cdutil.region.domain(longitude=(-125,-111),latitude=(42,49))
#Southwest (SW): (124°W–102°W, 31°N–42°N)
NCA4regions["SW"]=cdutil.region.domain(longitude=(-124,-102),latitude=(31,42))
#Upper Great Plains (GPu): (116°W–95°W, 40°N–49°N)
NCA4regions["GPu"]=cdutil.region.domain(longitude=(-116,-95),latitude=(40,49))
#Lower Great Plains (GPl): (107°W–93°W, 26°N–40°N)
NCA4regions["GPl"]=cdutil.region.domain(longitude=(-107,-93),latitude=(26,40))
#Midwest (MW): (97°W–80°W, 36°N–50°N)
NCA4regions["MW"]=cdutil.region.domain(longitude=(-97,-80),latitude=(36,50))
#Northeast (NE): (82°W–67°W, 37°N–48°N)
NCA4regions["NE"]=cdutil.region.domain(longitude=(-82,-67),latitude=(37,48))
#Southeast (SE): (95°W–76°W, 25°N–39°N)
NCA4regions["SE"]=cdutil.region.domain(longitude=(-95,-76),latitude=(25,39))


# In[3]:


def pull_data(curr_mod,curr_var,experiment_id,member_id,overwrite=False):
    # Baseline directory
    base_dir   = 'http://mary.ldeo.columbia.edu:81/CMIP6/.'
    # Write directory
    
    base_write_dir= external_drive+'DROUGHT/DOWNLOADED_RAW/'
    df_proclist = pd.DataFrame(columns=['model','sim','ensemble','variable'])
    ingrid_cmip6 = pd.read_csv("~/mary_cmip6.csv")


    write_dir = base_write_dir+curr_var+"/"+curr_mod+"/"
    write_stem = curr_var+"."+experiment_id+"."+curr_mod+"."+member_id+".*.nc"
    
    #If the directory doesn't exist already, make it
    os.makedirs(os.path.join(base_write_dir, curr_var, curr_mod),exist_ok=True)
    
    if not overwrite:
        already_exist=glob.glob(write_dir+write_stem)
        if len(already_exist)!=0:
            #print("Already done!")
            return False
    #rips=np.unique(np.array(df1.member_id))

    df1 = ingrid_cmip6[(ingrid_cmip6.source_id==curr_mod)                                   & (ingrid_cmip6.variable_id==curr_var) &                                    (ingrid_cmip6.experiment_id == experiment_id)&                                  (ingrid_cmip6.member_id == member_id)]

    #Construct openDAP link
    nfiles,nidentifiers=df1.shape
    times=np.sort(np.array(df1.time_range))
    i_ens=np.where(df1.time_range==times[0])[0]
    time_range=times[0]

    for time_range in times:
        # Construct Remote OpenDAP Link
        i_ens=np.where(df1.time_range==time_range)[0][0]
        nc_link = base_dir+df1.activity_id.iloc[i_ens]+'/.'+df1.institution_id.iloc[i_ens]+'/.'+curr_mod+'/.'+experiment_id+'/.'+df1.member_id.iloc[i_ens]+'/.'+df1.table_id.iloc[i_ens]+                             '/.'+curr_var+'/.'+df1.grid_label.iloc[i_ens]+'/.'+df1.version.iloc[i_ens]+'/.'+df1.file_basename.iloc[i_ens]+'/.'+curr_var+'/dods'
        request = requests.get(nc_link)
        if request.status_code == 200:
            #Get the data
            f=cdms.open(nc_link)
            data=f(curr_var)
            tax=data.getTime()
            tax.id="time"
            latax=data.getLatitude()
            lonax=data.getLongitude()
            #reshape it to years and months
            ntime=data.shape[0]
            nyears=int(ntime/12)
            rdata=data.reshape((nyears,12)+data.shape[1:])
            for i in range(nyears):
                yeardata=rdata[i]
                #Make the time axis
                tax_trunc=cdms.createAxis(tax[12*i:12*(i+1)])
                tax_trunc.designateTime()
                for key in tax.attributes.keys():
                    setattr(tax_trunc,key,tax.attributes[key])
                yeardata.setAxis(0,tax_trunc)
                yeardata.setAxis(1,latax)
                yeardata.setAxis(2,lonax)
                #get the start year for labeling purposes
                year=str(tax_trunc.asComponentTime()[0].year)
                writename = curr_var+"."+experiment_id+"."+curr_mod+"."+member_id+"."+year.zfill(4)+".nc"
                fw=cdms.open(write_dir+writename,"w")
                fw.write(yeardata)
                fw.close()
            f.close()
        return True


# In[4]:


def check_availability(curr_mod,curr_var,experiment_id):
    ingrid_cmip6 = pd.read_csv("~/mary_cmip6.csv")
    df1 = ingrid_cmip6[(ingrid_cmip6.source_id==curr_mod)                       & (ingrid_cmip6.variable_id==curr_var)                       & (ingrid_cmip6.experiment_id == experiment_id)]
    return(df1)
def get_members(curr_mod,curr_var,experiment_id):
    df1=check_availability(curr_mod,curr_var,experiment_id)
    return(np.unique(df1.member_id))


# In[5]:


def check_fixed_var_availability(curr_mod,fixed_var="sftlf",member_id="r1i1p1f1"):
    ingrid_cmip6 = pd.read_csv("~/mary_cmip6.csv")
    df1 = ingrid_cmip6[(ingrid_cmip6.source_id==curr_mod)                       & (ingrid_cmip6.variable_id==fixed_var)                       & (ingrid_cmip6.member_id == member_id)]
    return(df1)


# In[6]:


def pull_fixedvar(curr_mod,curr_var,experiment_id,member_id,overwrite=False):

    # Baseline directory
    base_dir   = 'http://mary.ldeo.columbia.edu:81/CMIP6/.'
    # Write directory
    
    base_write_dir= external_drive+'DROUGHT/fixedvar/'
    df_proclist = pd.DataFrame(columns=['model','sim','ensemble','variable'])
    ingrid_cmip6 = pd.read_csv("~/mary_cmip6.csv")


    write_dir = base_write_dir
    write_stem=curr_var+"_fx_"+curr_mod+".nc"
    #write_stem = curr_var+"."+experiment_id+"."+curr_mod+"."+member_id+".*.nc"
    
   
    if not overwrite:
        already_exist=glob.glob(write_dir+write_stem)
        if len(already_exist)!=0:
            print("already exists")
            #return False
    #rips=np.unique(np.array(df1.member_id))

    df1 = ingrid_cmip6[(ingrid_cmip6.source_id==curr_mod)                                   & (ingrid_cmip6.variable_id==curr_var) &                                    (ingrid_cmip6.experiment_id == experiment_id)&                                  (ingrid_cmip6.member_id == member_id)]

    #Construct openDAP link
    nfiles,nidentifiers=df1.shape
    if nfiles==1:
        i_ens=0
        nc_link = base_dir+df1.activity_id.iloc[i_ens]+'/.'+df1.institution_id.iloc[i_ens]+'/.'+curr_mod+'/.'+experiment_id+'/.'+df1.member_id.iloc[i_ens]+'/.'+df1.table_id.iloc[i_ens]+                             '/.'+curr_var+'/.'+df1.grid_label.iloc[i_ens]+'/.'+df1.version.iloc[i_ens]+'/.'+df1.file_basename.iloc[i_ens]+'/.'+curr_var+'/dods'
        request = requests.get(nc_link)
        if request.status_code == 200:
            #Get the data
            f=cdms.open(nc_link)
            data=f(curr_var)
            fw=cdms.open(write_dir+write_stem,"w")
            fw.write(data)
            fw.close()
    else:
        print("more than one candidate found")
        print(df1)


# In[7]:


def pull_land_fractions():
    models=[x.split("/")[-1] for x in glob.glob(external_drive+"DROUGHT/DOWNLOADED_RAW/tas/*")]
    already_done=[x.split("_fx_")[-1].split(".")[0] for x in glob.glob(external_drive+"DROUGHT/fixedvar/sftlf*")]
    not_yet=np.setdiff1d(models,already_done)


    for model in not_yet:
        print(model)
      #  print(len(check_fixed_var_availability(model)))

        if "piControl" in np.array(check_fixed_var_availability(model).experiment_id):
            pull_fixedvar(model,"sftlf","piControl","r1i1p1f1")
        elif "amip" in np.array(check_fixed_var_availability(model).experiment_id):
            pull_fixedvar(model,"sftlf","amip","r1i1p1f1")
        else:
            print("nothing found for ",model)


# In[8]:


def download_hydrological_data(variables,experiments,models=None,verbose=False):
    ingrid_cmip6 = pd.read_csv("~/mary_cmip6.csv")
    if models is None:
        models=np.unique(ingrid_cmip6.source_id)
    for experiment_id in experiments:
        for variable in variables:
            for model in models:
                rips=get_members(model,variable,experiment_id)
                for rip in rips:
                    the_direc="/home/kdm2144/DROUGHT/DOWNLOADED_RAW/"+variable+"/"+model+"/"
                    num_already_downloaded=len(glob.glob(the_direc+"*"+experiment_id+"."+model+"."+rip+"*"))
                    
                    if num_already_downloaded==0:
                        try:
                            downloaded=pull_data(model,variable,experiment_id,rip)
                            if verbose:
                                if downloaded:
                                    print("model: ",model)
                                    print("variable: ",variable)
                                    print("experiment: ",experiment)
                                    print("rip: ",rip)
                        except: 
                            if verbose:
                                print("DID NOT DOWNLOAD:")
                                print("model: ",model)
                                print("variable: ",variable)
                                print("experiment: ",experiment_id)
                                print("rip: ",rip)
                    else:
                        if verbose:
                            print("Already downloaded!")


def NCA4_regions_average(variables,experiments,overwrite=False):

    datadirec=external_drive+"DROUGHT/DOWNLOADED_RAW/"
    writedirec=external_drive+"DROUGHT/NCA4/"
    
    fixedvardirec=external_drive+"DROUGHT/fixedvar/"



    ###### LOOP OVER ALL VARIABLES #####
    for variable in variables:

        for region in NCA4regions.keys():
            #cmd = "mkdir "+writedirec+"/"+region+"/"+variable
            os.makedirs(writedirec+"/"+region+"/"+variable,exist_ok=True)


        
        modeldirs=glob.glob(datadirec+variable+"/*")
        ### LOOP OVER ALL MODELS
        for direc in modeldirs:
            model=direc.split("/")[-1]
            
            allfiles=glob.glob(direc+"/*"+variable+"*")
            

            ###### LOOP OVER ALL EXPERIMENTS #####
            for experiment in experiments:
                writedirecs={}
                for region in NCA4regions.keys():
                
                    region_writedirec=writedirec+region+"/"+variable+"/"+experiment+"/"
                    cmd="mkdir "+region_writedirec
                    os.system(cmd)
                    writedirecs[region]=region_writedirec

                allfiles_experiment=glob.glob(direc+"/"+variable+"."+experiment+".*")
                rips=np.unique([x.split(".")[-3] for x in allfiles_experiment])

                landthresh=1
                #Get the land fraction
                landfiles=glob.glob(fixedvardirec+"sftlf*"+model+".*")


                if len(landfiles)==1:
                    fland=cdms.open(landfiles[0])
                    landfrac=fland("sftlf")
                    fland.close()
                else:
                    print("can't find land fraction file for", model)
                    print(landfiles)
                    continue
                #print(direc)



                ###### LOOP OVER ALL RIPS #####
                for rip in rips:
                    #print(rip)
                    writenames={}
                    for key in NCA4regions.keys():
                        writename=writedirecs[key]+variable+"."+experiment+"."+model+"."+rip+".nc"

                        writenames[key]=writename

                    yearcheck=[]
                    
                    ripfiles=np.sort(glob.glob(direc+"/"+variable+"."+experiment+"."+model+"."+rip+"*"))
                    
                    L=len(ripfiles)
                    i=0
                    ripfile=ripfiles[i]
                    frip=cdms.open(ripfile)
                    data=frip(variable)
                    frip.close()
                    if data.shape[1:]!=landfrac.shape:
                        print("land mask wrong shape for "+variable+"."+experiment+"."+model+"."+rip)
                        continue
                    latax=landfrac.getLatitude()
                    lonax=landfrac.getLongitude()
                    tax=np.arange(12)
                    fpath,fexpt,fmodel,frip,fyear,fnc=ripfile.split(".")
                    ###Loop over regions
                    for key in NCA4regions.keys():
                        if writenames[key] in glob.glob(writedirecs[key]+"*"):
                            if not overwrite:
                                continue
                        if key=="NW":
                            print(variable+"."+experiment+"."+model+"."+rip+"*")
                        region=NCA4regions[key]
                        DATA=MV.zeros(L*12)
                        
                        landdata=cmip5.cdms_clone(np.repeat(.01*landfrac.asma()[np.newaxis],12,axis=0)*data,data)
                        landdata.setAxis(1,latax)
                        landdata.setAxis(2,lonax)
                       

                       
                        for i in range(L):

                            ripfile=ripfiles[i]

                            f=cdms.open(ripfile)
                            data=f(variable)

                           
                            landdata=cmip5.cdms_clone(np.repeat(.01*landfrac.asma()[np.newaxis],12,axis=0)*data,data)
                            #Kludge since downloading process didn't preserve lat/lon designation
                            landdata.setAxis(1,latax)
                            landdata.setAxis(2,lonax)
                            f.close()

                            fpath,fexpt,fmodel,frip,fyear,fnc=ripfile.split(".")
                            DATA[12*i:12*(i+1)]=cdutil.averager(landdata(region),axis='xy')
      
      
                            yearcheck+=[float(fyear)]
                        tax=cdms.createAxis(np.arange(L*12))
                        tax.designateTime()
                        tax.units='months since '+str(yearcheck[0])+'-1-1'
                        if variable in ["pr","prsn","mrros","mrro","evspsbl"]:
                            DATA=DATA*60*60*24 #convert to mm day-1
                            DATA.units="mm day -1"
                        else:
                            DATA.units="kg m-2"
                        DATA.setAxis(0,tax)
                        DATA.id=variable
                        fw=cdms.open(writenames[key],"w")
                        fw.write(DATA)
                        fw.close()
                        #if key == "SE":
                            #print(variable+"."+experiment+"."+model+"."+rip+"*")
    


# In[ ]:




