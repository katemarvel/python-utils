import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# cartopy stuff
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry.polygon import LinearRing
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Map Stuff
# Some mapping variables
def plot_CONUS(mme_diff,cmap_name=plt.cm.BrBG,vmax=None,vmin=None):
    lono=mme_diff.getLongitude()[:]
    lato=mme_diff.getLatitude()[:]
    lon,lat=np.meshgrid(lono,lato)
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
            name='admin_1_states_provinces_lines',\
            scale='50m',\
            facecolor='none')
    extent_lonlat = (-125, -70, 22, 50)

    #clevs = np.array([-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,.3,.4,.5,.6])*2.5
    if vmin is None:
        vmin=np.percentile(mme_diff.compressed(),5)
    if vmax is None:
        vmax=np.percentile(mme_diff.compressed(),95)
    clevs=np.linspace(vmin,vmax,13)
    clevs_units=clevs.copy()
    nmap = plt.cm.get_cmap(name=cmap_name,lut=clevs.size-1)

    ocean_color = np.float64([209,230,241])/255

    fig = plt.figure(figsize=(12, 12),facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=-100, central_latitude=25, globe=None))
    m = ax.contourf(lon, lat, mme_diff,clevs,transform=ccrs.PlateCarree(),cmap=nmap,extend="both")
    ax.coastlines()
    ax.set_global()
    ax.set_extent(extent_lonlat, crs=ccrs.PlateCarree())
    #ax.gridlines(xlocs=np.arange(-180,190,10),ylocs=np.arange(-180,190,10))
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='-', edgecolor='k')
    ax.add_feature(states_provinces, linewidth=0.5, linestyle='-', edgecolor='k')
    #ax.add_feature(newcoast, linewidth=0.5, linestyle='-', edgecolor='k')
    #ax.add_feature(newlake, linewidth=0.5, linestyle='-', edgecolor='k')
    ax.add_feature(cartopy.feature.LAND,color='w',zorder=0,edgecolor='k')
    ax.add_feature(cartopy.feature.OCEAN,color=ocean_color,zorder=0,edgecolor='k')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='-', edgecolor='k')
    ax.add_feature(cartopy.feature.OCEAN,color=ocean_color,zorder=1)
    #ax.add_feature(newcoast, linewidth=1, linestyle='-', zorder=2,edgecolor='k')
    #ax.text(-122,21,var_txt+' ('+seas_txt+')',transform=ccrs.PlateCarree(),fontsize=32,fontweight="bold", \
     #horizontalalignment='center', verticalalignment='center',)
    #ax.text(-122,17,ssp_txt,transform=ccrs.PlateCarree(),fontsize=28,fontweight="normal", \
     #horizontalalignment='center', verticalalignment='center',)
    cbar=plt.colorbar(m,orientation="horizontal",fraction=0.08,pad=0.04,ticks=clevs_units[np.arange(0,clevs_units.size+1,2)])
    cbar.ax.tick_params(labelsize=24)
