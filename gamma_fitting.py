import pygrib
import numpy as np
from scipy.integrate import quadrature as quad
from scipy.stats import gamma

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

def cdf(x, data):
    return np.interp(x, np.linspace(0.01, 0.99, 99), data)

def approx_gamma_params(data):
    loc = np.where(data != 0)[0].min() + 2
    def cdf(x):
        return cdf(x, data)
    def cdf_shift(x):
        return 1-cdf(x)
    def x_cdf_shift(x):
        return x*(1-cdf(x))
    data_max = cdf(.99)
    mom1, _ = quad(cdf_shift, loc, data_max)
    mom2, _ = quad(x_cdf_shift, loc, data_max)
    mean = mom1
    var = mom2 - mom1**2
    shape = mean**2/var
    scale = mean/var
    # return shape, loc, scale
    return gamma.ppf([0.30, 0.60, 0.90], shape, loc=loc, scale=scale)

# initializing output
level_width = 30 # level_width defined here, but any updates should be manually made in wrap function
# global precip_levels_approx_var
precip_levels_approx_gamma = np.zeros(shape=(int(np.floor(100/level_width)),)+lat.shape)
# global nonzero_idx
nonzero_idx = np.where(precip_levels[-1,:,:] != 0)

# wrapped function for parallel processing
def wrap(n):
    i = nonzero_idx[0][n]
    j = nonzero_idx[1][n]
    precip_levels_approx_gamma[:,i,j] = approx_gamma_params(precip_levels[:,i,j])

# parallel code using multiprocessing - doesn't seem to speed up code with 8 cores though!
if __name__ == '__main__':
    if mp.cpu_count() > 16:
        pool = mp.Pool(processes = mp.cpu_count()-16)
    else:
        pool = mp.Pool(processes = np.cpu_count())
    pool.map_async(wrap, list(range(nonzero_idx[0].shape[0])))
    pool.close()
    pool.join()

np.save('precip_levels_approx_gamma', precip_levels_approx_gamma)