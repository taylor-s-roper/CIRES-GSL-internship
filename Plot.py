import numpy as np
import pygrib

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap, cm


fn_grb = 'blend.t00z.qmd.f012.co.grib2'
ds_grb = pygrib.open(fn_grb)
lat, long = ds_grb.message(2).data()[1:]
precip_shape = lat.shape
precip_levels = np.zeros(shape=(99,)+precip_shape)
for i in range(99):
    precip_levels[i,:,:] = ds_grb.message(i+2).data()[0]

precip_levels_approx_unif = np.load('precip_levels_approx_unif.npy')
precip_levels_approx_var = np.load('precip_levels_approx_var_.npy')

class Plot:

    def __init__(self):
        self.lat = lat
        self.long = long
        self.orig = precip_levels
        self.approx_unif = precip_levels_approx_unif
        self.approx_var = precip_levels_approx_var

    def Basemap_plot(self, level, diff, unif, scale=1):

        data = self.orig[level-1,:,:]

        possible_levels = np.array([30, 60, 90])
        level_idx = np.where(possible_levels == level)[0][0]
        if diff:
            if unif:
                data = self.approx_unif[level_idx,:,:]-data
                var_name = f'Uniform node error at {level}% level'
            else:
                data = self.approx_var[level_idx,:,:]-data
                var_name = f'Variable node error at {level}% level'
        else:
            var_name = f'Precipiation at probability level {level}%'

        
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

        x, y = map(self.long, self.lat)

        if diff:
            if scale ==0:
                data_abs_max = int(np.ceil(max(np.abs(data.min()),np.abs(data.max()))))
                levels = list(range(-data_abs_max,data_abs_max+1))
            else:
                levels = list(range(-scale, scale+1))
            plt.pcolormesh(x, y, data,
                        norm=colors.Normalize(vmin=levels[0], vmax=levels[-1]),
                        cmap='seismic', shading='nearest')
            # map.contourf(x, y, data, 16, levels=levels, cmap='seismic')
            map.colorbar()
            map.colorbar().set_label('mm of precipitation')
        else:
            map.contourf(x, y, data, 16, linewidths=1.5)

        plt.title(var_name)
        plt.show()