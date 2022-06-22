import numpy as np
import pygrib
from scipy import interpolate, optimize 
import multiprocessing as mp

def linear_splines_unif(data, num_knots=10, level_width=1):    
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
    knot_vals = list(params[0][0:num_knots])
    knots = list(params[0][num_knots:])
    return np.interp(x, knots, knot_vals)

def linear_splines_var(data, num_knots, level_width):
    if data[-1] == 0:
        count[1] += 1
        return np.zeros(int(np.floor(100/level_width)))
    p_0 = np.linspace(0,98,num_knots).astype(int)
    p_0 = np.hstack([data[p_0], p_0])
    try:
        fit, _ = optimize.curve_fit(lambda x, *params : linear_splines(x, num_knots, params),
                                    np.linspace(1,99,99), data, p_0)
        count[2] += 1
        levels = np.linspace(1,99,99)
        levels = levels[level_width-1::level_width]
        return np.interp(levels, fit[num_knots:], fit[:num_knots])
    except RuntimeError:
        count[0] += 1
        return linear_splines_unif(data, num_knots=num_knots, level_width=level_width)

fn_grb = 'blend.t00z.qmd.f012.co.grib2'
ds_grb = pygrib.open(fn_grb)
lat, long = ds_grb.message(2).data()[1:]
precip_shape = lat.shape
global precip_levels 
precip_levels = np.zeros(shape=(99,)+precip_shape)
for i in range(99):
    precip_levels[i,:,:] = ds_grb.message(i+2).data()[0]

count = [0,0,0]
level_width = 30
global precip_levels_approx_var
precip_levels_approx_var = np.zeros(shape=(int(np.floor(100/level_width)),)+lat.shape)

def wrap(row):
    i = row
    for j in range(lat.shape[1]):
        precip_levels_approx_var[:,i,j] = linear_splines_var(precip_levels[:,i,j],10,30)

if __name__ == '__main__':
    pool = mp.Pool(processes = 16)
    pool.map_async(wrap, np.array(range(lat.shape[0])))
    pool.close()
    pool.join()

np.save('precip_levels_approx_var_', precip_levels_approx_var)