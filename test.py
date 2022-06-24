import numpy as np
import pygrib
from scipy import interpolate
import multiprocessing as mp
import time

def linear_splines_unif(data, num_knots=10, level_width=1):   
    '''
    Calculates piecewise linear splines for quantile data using specified number of 
    knots uniformly spaced and returning interpolated approximation at every 
    quantile level with level_width.
    ''' 

    # dealing with all 0 data
    if data[-1] == 0:
        return np.zeros(int(np.floor(100/level_width)))

    # calculating where cdf starts being nonzero 
    levels = np.array(range(1,100))
    knot_ = np.where(data > 0)[0].min() - 1  

    # constructing knots
    if knot_ > 1:
        knots = np.unique(np.linspace(knot_-1, 98, num_knots-1, dtype=int))
        knots = np.insert(knots, 0, 0)
    else:
        knots = np.unique(np.linspace(0, 98, num_knots, dtype=int))
        
    # approx = interpolate.interp1d(knots+1, data[knots], assume_sorted=True) 
    # return approx(levels[level_width-1::level_width])
    return np.interp(levels[level_width-1::level_width], knots+1, data[knots])

# read in grib file
fn_grb = 'blend.t00z.qmd.f012.co.grib2'
ds_grb = pygrib.open(fn_grb)

# latitude and longitude grid values
lat, long = ds_grb.message(2).data()[1:]

# extracting precipitation levels for 6 hr forecast
precip_shape = lat.shape
precip_levels = np.zeros(shape=(99,)+precip_shape)
for i in range(99):
    precip_levels[i,:,:] = ds_grb.message(i+2).data()[0]

# flattening precip_levels
global precip_levels_flat
precip_levels_flat = np.zeros(shape=(99,lat.shape[0]*lat.shape[1]))

for element in range(lat.shape[0]*lat.shape[1]):
    i = int(np.floor(element/lat.shape[1]))
    j = int(element % lat.shape[1])
    precip_levels_flat[:,element] = precip_levels[:,i,j]

# looping
start_time = time.time()

# input values
num_knots = 10
level_width = 30

# initializing output 
precip_levels_approx_ = np.zeros(shape=(int(np.floor(100/level_width)),)+lat.shape)

# looping through grid
for element in range(lat.shape[0]*lat.shape[1]):
    i = int(np.floor(element/lat.shape[1]))
    j = int(element % lat.shape[1])
    precip_levels_approx_[:,i,j] = linear_splines_unif(precip_levels_flat[:,element],10,30)

end_time = time.time()
print(f'Linear spline code with loop took {end_time - start_time} seconds to run.')

start_time = time.time()

# initializing output 
global precip_levels_approx
precip_levels_approx = np.zeros(shape=(int(np.floor(100/level_width)),)+lat.shape)

# parallel code using multiprocessing - doesn't speed up code though!
def wrap(element):
    i = int(np.floor(element/lat.shape[1]))
    j = int(element % lat.shape[1])
    precip_levels_approx[:,i,j] = linear_splines_unif(precip_levels_flat[:,element],10,30)

if __name__ == '__main__':
    pool = mp.Pool(processes = mp.cpu_count())
    pool.map_async(wrap, list(range(lat.shape[0]*lat.shape[1])))
    pool.close()
    pool.join()

end_time = time.time()
print(f'Parallel linear spline code took {end_time - start_time} seconds to run.')

#np.save('precip_levels_approx_var_', precip_levels_approx_var)
