import numpy as np
import pygrib
from scipy import interpolate, optimize 
import multiprocessing as mp
import time

# linear spline functions

def linear_splines_unif(data, num_knots=10, level_width=1):   
    '''
    Calculates piecewise linear splines for quantile data using specified number of 
    knots uniformly spaced and returning interpolated approximation at every 
    quantile level with level_width.
    ''' 

    # calculating where cdf starts being nonzero (all zero cdf's should not be inputted)
    levels = np.array(range(1,100))
    knot_ = np.where(data > 0)[0].min() - 1   
    if knot_ > 1:
        knots = np.unique(np.linspace(knot_-1, 98, num_knots-1, dtype=int))
        knots = np.insert(knots, 0, 0)
    else:
        knots = np.unique(np.linspace(0, 98, num_knots, dtype=int))
        
    # approx = interpolate.interp1d(knots+1, data[knots], assume_sorted=True) 
    # return approx(levels[level_width-1::level_width])
    return np.interp(levels[level_width-1::level_width], knots+1, data[knots])

def linear_splines(x, num_knots, *params):
    '''
    Function to be used in scipy.optimize.curve_fit in linear_splines_var function.
    '''

    knot_vals = list(params[0][0:num_knots])
    knots = list(params[0][num_knots:])
    return np.interp(x, knots, knot_vals)

def linear_splines_var(data, num_knots, level_width, all=True):
    '''
    Calculates piecewise linear splines for quantile data using specified number of
    knots with optimized placement and returning interpolated approximation at every
    quantile level with level_width.
    '''
    print(data.shape)
    if not all:
        if data[49] == 0:
            return linear_splines_unif(data, num_knots=num_knots, level_width=level_width) 

    # checking if cdf is all zero
    if data[-1] == 0:
        return np.zeros(int(np.floor(100/level_width)))

    # setting up intial value of parameters
    p_0 = np.linspace(0,98,num_knots).astype(int)
    p_0 = np.hstack([data[p_0], p_0])

    # try to fit parameters with RuntimeError exception that returns linear_splines_unif
    # that uses uniformly space knots
    try:
        fit, _ = optimize.curve_fit(lambda x, *params : linear_splines(x, num_knots, params), 
                np.linspace(1,99,99), data, p_0)
        levels = np.linspace(1,99,99)
        levels = levels[level_width-1::level_width]
        return np.interp(levels, fit[num_knots:], fit[:num_knots])
    except RuntimeError:
        return linear_splines_unif(data, num_knots=num_knots, level_width=level_width)

# wrapper function of linear_splines_var for numpy.apply_over_axes
def wrapper_func(data, axis):
    num_knots = 10
    level_width = 30
    all = False
    return linear_splines_var(data, num_knots, level_width, all)


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
# global precip_levels_flat
precip_levels_flat = np.zeros(shape=(99,lat.shape[0]*lat.shape[1]))

for element in range(lat.shape[0]*lat.shape[1]):
    i = int(np.floor(element/lat.shape[1]))
    j = int(element % lat.shape[1])
    precip_levels_flat[:,element] = precip_levels[:,i,j]

# initializing output 
# global precip_levels_approx_var
# precip_levels_approx_var = np.zeros(shape=(int(np.floor(100/level_width)),)+lat.shape)

# parallel code using multiprocessing - doesn't speed up code though!
# def wrap(element):
#     i = int(np.floor(element/lat.shape[1]))
#     j = int(element % lat.shape[1])
#     precip_levels_approx_var[:,i,j] = linear_splines_var(precip_levels_flat[:,element],10,30)

# if __name__ == '__main__':
#     pool = mp.Pool(processes = mp.cpu_count())
#     pool.map_async(wrap, list(range(lat.shape[0]*lat.shape[1])))
#     pool.close()
#     pool.join()

start_time = time.time()

# applying linear_splines_var to every grid point
precip_levels_approx_var = np.apply_over_axes(wrapper_func, precip_levels_flat, 0)

end_time = time.time()
print(f'Linear spline code took {end_time - start_time} to run.')

np.save('precip_levels_approx_var_', precip_levels_approx_var)
