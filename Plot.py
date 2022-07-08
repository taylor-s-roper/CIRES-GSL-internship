month = 'wet'
level = 50
plot_errors = False

import numpy as np
import pygrib

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap, cm

# read in grib file

if month == 'jan':
    date = '20220111'
    lead_time = '00'
    forecast_time = '018'
elif month == 'march':
    date = '20220324'
    lead_time = '12'
    forecast_time = '018'
elif month == 'april':
    date = '20220425'
    lead_time = '12'
    forecast_time = '018'
elif month == 'july':
    date = '20220703'
    lead_time = '00'
    forecast_time = '018'
elif month == 'sept':
    date = '20210920'
    lead_time = '12'
    forecast_time = '018'
elif month == 'oct':
    date = '20211015'
    lead_time = '00'
    forecast_time = '018'
elif month == 'dec':
    date = '20211205'
    lead_time = '12'
    forecast_time = '018'
elif month == 'wet':
    date = ''
    lead_time = '00'
    forecast_time = '012'
else:
    print('Issue with month input.')
fn_grb = 'data/blend' + date + '.t' + lead_time + 'z.qmd.f' + forecast_time + '.co.grib2'
ds_grb = pygrib.open(fn_grb)

# latitude and longitude grid values
lat, long = ds_grb.message(2).data()[1:]

# extracting data
precip_shape = lat.shape
precip = np.zeros(shape=(99,)+precip_shape)
for i in range(99):
    precip[i,:,:] = ds_grb.message(i+2).data()[0] # ACPC:surface:12-18 hour acc fcst
ds_grb.close()

precip_unif = np.load('results/precip_unif_' + month, allow_pickle=True)
precip_var = np.load('results/precip_var_' + month, allow_pickle=True)
errors_unif = np.load('results/errors_unif_' + month, allow_pickle=True)
errors_var = np.load('results/errors_var_' + month, allow_pickle=True) 


def Basemap_plot(data, long, lat, diff=False, name=None, color_label='mm of precipiation'):
                
    map = Basemap(llcrnrlon=-123.,llcrnrlat=20., 
                urcrnrlon=-59., urcrnrlat=48., 
                projection='lcc', 
                lat_1=38.5,
                lat_0=38.5,
                lon_0=-97.5,
                resolution='l')

    # draw coastlines, country boundaries, fill continents
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='xkcd:white',lake_color='xkcd:white')



    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='xkcd:white')
    map.drawstates()

    # draw lat/lon grid lines every 30 degrees.
    map.drawmeridians(np.arange(-180,180,30))
    map.drawparallels(np.arange(-90,90,30))

    x, y = map(long, lat)

    if diff:
        boundary = int(np.ceil(max(np.abs(data.min()), np.abs(data.max()))))
        levels = list(range(-boundary, boundary+1))
        plt.pcolormesh(x, y, data,
                    norm=colors.Normalize(vmin=levels[0], vmax=levels[-1]),
                    cmap='seismic', shading='nearest')
        # map.contourf(x, y, data, 16, levels=levels, cmap='seismic')
        map.colorbar()
        map.colorbar().set_label(color_label)
    else:
        map.contourf(x, y, data, 16, linewidths=1.5)
        map.colorbar()
        map.colorbar().set_label(color_label)
        
    if name is not None:
        plt.title(name)
        
    plt.show()

level_idx = np.where(np.array([5,25,50,75,95]) == level)[0]
Basemap_plot(data=precip[level-1,:,:], long=long, lat=lat, name=f'Precipitation at {level}% quantile')
Basemap_plot(data=precip_unif[level-1,:,:]-precip[level-1,:,:], long=long, lat=lat, diff=True, name=f'Uniform node error at {level}% quantile')
Basemap_plot(data=precip_var[level-1,:,:]-precip[level-1,:,:], long=long, lat=lat, diff=True, name=f'Variable node error at {level}% quantile')
if plot_errors:
    Basemap_plot(data=errors_unif[0,:,:], long=long, lat=lat, name=f'KS statistic for uniform nodes at {level}% quantile', color_label='KS')
    Basemap_plot(data=errors_unif[1,:,:], long=long, lat=lat, name=f'L_1 norm for uniform nodes at {level}% quantile', color_label='L_1 norm')
    Basemap_plot(data=errors_unif[2,:,:], long=long, lat=lat, name=f'CRPS for uniform nodes at {level}% quantile', color_label='CRPS')
    Basemap_plot(data=errors_var[0,:,:], long=long, lat=lat, name=f'KS statistic for variable nodes at {level}% quantile', color_label='KS')
    Basemap_plot(data=errors_var[1,:,:], long=long, lat=lat, name=f'L_1 norm for variable nodes at {level}% quantile', color_label='L_1 norm')
    Basemap_plot(data=errors_var[2,:,:], long=long, lat=lat, name=f'CRPS for variable nodes at {level}% quantile', color_label='CRPS')