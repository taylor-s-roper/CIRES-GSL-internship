approx_type = 'both' # 'unif', 'var', or 'both'
include_temp = False # does not approximate temperature data yet
error_calc = True
save = True
month = 'june'
include_obs = True
# months completed: june 

import numpy as np
import pygrib
from scipy import optimize

import time

time_start = time.time()

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
elif month == 'june':
    date = '20220626'
    lead_time = '12'
    forecast_time = '006'
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
    precip[i,:,:] = ds_grb.message(i+2).data()[0] 
ds_grb.close()

if include_temp:
    ds_grb = pygrib.open(fn_grb)
    temp_shape = lat.shape
    temp = np.zeros(shape=(99,)+temp_shape)
    for i in range(99):
        temp[i,:,:] = ds_grb.message(i+215).data()[0]
    ds_grb.close()

if include_obs:
    fn_grb = 'data/urma2p5.2022062612.pcp_06h.wexp.grb2'
    ds_grb = pygrib.open(fn_grb)
    obs = ds_grb.message(1).data()[0]
    ds_grb.close()

# masking precip at grid points that are not monotonic
mask = np.zeros(precip.shape)
for i in range(lat.shape[0]):
    for j in range(lat.shape[1]):
        issue = False
        for level in range(99-1):
            if precip[level,i,j] > precip[level+1,i,j]:
                if not issue:
                    issue = True
                    #mask[level+1:,i,j] = np.ones(mask.shape[0] - level - 1)
                    mask[:,i,j] = np.ones(mask.shape[0])
                    
precip = np.ma.masked_array(precip, mask)
if include_obs:
    obs = np.ma.masked_array(obs, mask[-1,:,:])


levels = np.linspace(1,99,99)
qs = levels / 100
N = int(1e3)
if include_obs:
    data_max = max(obs.max(), precip[-1,:,:].max())
else:
    data_max = precip[-1,:,:].max()
xs = np.linspace(0.0, data_max, N)

# linear spline and error functions

def linear_splines_unif(data, num_knots=10, zero_inflated=True):   
    '''
    Calculates piecewise linear splines for quantile data using specified number of 
    knots uniformly spaced and returning interpolated approximation at every 
    quantile level.
    ''' 

    # checking if cdf is all zero
    #if data[-1] == 0:
    #    return np.zeros(99)
    
    if zero_inflated:
        # calculating where cdf starts being nonzero (all zero cdf's should not be inputted)
        knot_ = np.where(data > 0)[0].min() - 1   
        if knot_ > 1:
            knots = np.unique(np.linspace(knot_-1, 98, num_knots-1, dtype=int))
            knots = np.insert(knots, 0, 0)
        else:
            knots = np.unique(np.linspace(0, 98, num_knots, dtype=int))
    else:
        knots = np.unique(np.linspace(0, 98, num_knots, dtype=int))
        
    levels = range(1,100)
    return np.interp(levels, knots+1, data[knots])

def linear_splines(x, num_knots, *params):
    '''
    Function to be used in scipy.optimize.curve_fit in linear_splines_var function.
    '''

    knot_vals = list(params[0][0:num_knots])
    knots = list(params[0][num_knots:])
    return np.interp(x, knots, knot_vals)

def linear_splines_var(data, num_knots=5, zero_inflated=True):
    '''
    Calculates piecewise linear splines for quantile data using specified number of
    knots with optimized placement and returning interpolated approximation at every
    quantile level with level_width.
    '''

    # checking if cdf is all zero
    #if data[-1] == 0:
    #    return np.zeros(99)
    
    data_ = data # saving full set of data in case optimize.curve_fit fails
    
    if zero_inflated:
        # calculating where cdf starts being nonzero (all zero cdf's should not be inputted)
        idx_start = max(np.where(data > 0)[0].min() - 1, 0)
        data = data[idx_start:]
        if 99-idx_start < num_knots*2:
            return linear_splines_unif(data_, num_knots=5, zero_inflated=zero_inflated)
    else:
        idx_start = 0
        
    # setting up intial value of parameters
    p_0 = np.linspace(idx_start,98,num_knots)
    p_0 = np.hstack([np.interp(p_0, levels[idx_start:], data), p_0])

    # try to fit parameters with RuntimeError exception that returns linear_splines_unif
    # that uses uniformly space knots
    try:
        fit, _ = optimize.curve_fit(lambda x, *params : linear_splines(x, num_knots, params), 
                np.array(range(idx_start+1,100)), data, p_0)
        levels_ = range(idx_start+1,100)
        for k in range(num_knots-1):
            if fit[:num_knots][k+1] < fit[:num_knots][k]:
                return linear_splines_unif(data_, num_knots, zero_inflated)
        return np.hstack([np.zeros(idx_start), np.interp(levels_, fit[num_knots:], fit[:num_knots])])
    except RuntimeError:
        return linear_splines_unif(data_, num_knots, zero_inflated)

def calc_errors(orig, approx):
    if orig[-1] == 0 and approx[-1] == 0:
        return np.zeros(3)
    else:
        differences = np.abs(np.interp(xs[1:-1], orig[1:-1], qs[1:-1], left=0.0, right=1.0) - np.interp(xs[1:-1], approx[1:-1], qs[1:-1], left=0.0, right=1.0))
        return [differences.max(), np.mean(differences), np.mean(differences**2)] # [KS, mean L_1, CRPS]

def obs_CRPS(obs, approx):
    obs_cdf = np.zeros(N-2)
    obs_nonzero = np.where(xs[1:-1] >= obs)[0]
    obs_cdf[obs_nonzero] = np.ones(obs_nonzero.shape[0])
    return np.mean((np.interp(xs[1:-1], approx[1:-1], qs[1:-1], left=0.0, right=1.0) - obs_cdf)**2)

precip_unif = np.ma.masked_array(np.zeros(precip.shape), mask)
precip_var = np.ma.masked_array(np.zeros(precip.shape), mask)
if error_calc:
    errors_unif = np.ma.masked_array(np.zeros((3,) + lat.shape), [mask[-1,:,:],mask[-1,:,:],mask[-1,:,:]])
    errors_var = np.ma.masked_array(np.zeros((3,) + lat.shape), [mask[-1,:,:],mask[-1,:,:],mask[-1,:,:]])    
if include_obs:
    obs_crps = np.ma.masked_array(np.zeros((3,)+lat.shape), [mask[-1,:,:],mask[-1,:,:],mask[-1,:,:]])
idx = np.where(mask[-1,:,:] == 0)
level_idxs = [4, 24, 50, 74, 94]

if approx_type == 'unif':
    for n in range(idx[0].shape[0]):
        i = idx[0][n]
        j = idx[1][n]
        if precip[-1,i,j] != 0:
            precip_unif[:,i,j] = linear_splines_unif(precip[:,i,j])  
            if error_calc:
                errors_unif[:,i,j] = calc_errors(precip[:,i,j], precip_unif[:,i,j])
        if include_obs:
            obs_crps[0,i,j] = obs_CRPS(obs[i,j], precip[:,i,j])
            obs_crps[1,i,j] = obs_CRPS(obs[i,j], precip_unif[:,i,j])
    if save:
        precip_unif[level_idxs,:,:].dump('results/' + date + '/precip_unif')
        if error_calc:
            errors_unif.dump('results/' + date + '/errors_unif')
        if include_obs:
            obs_crps[0,:,:].dump('results/' + date + '/obs_crps')
            obs_crps[1,:,:].dump('results/' + date + '/obs_unif_crps')

elif approx_type == 'var':
    for n in range(idx[0].shape[0]):
        i = idx[0][n]
        j = idx[1][n]
        if precip[-1,i,j] != 0:
            precip_var[:,i,j] = linear_splines_var(precip[:,i,j]) 
            if error_calc:
                errors_var[:,i,j] = calc_errors(precip[:,i,j], precip_var[:,i,j])
        if include_obs:
            obs_crps[0,i,j] = obs_CRPS(obs[i,j], precip[:,i,j])            
            obs_crps[2,i,j] = obs_CRPS(obs[i,j], precip_var[:,i,j])    
    if save:
        precip_var[level_idxs,:,:].dump('results/' + date + '/precip_var')
        if error_calc:
            errors_var.dump('results/' + date + '/errors_var')
        if include_obs:
            obs_crps[0,:,:].dump('results/' + date + '/obs_crps')            
            obs_crps[2,:,:].dump('results/' + date + '/obs_var_crps')

elif approx_type == 'both':
    for n in range(idx[0].shape[0]):
        i = idx[0][n]
        j = idx[1][n]
        if precip[-1,i,j] != 0:
            precip_unif[:,i,j] = linear_splines_unif(precip[:,i,j])  
            precip_var[:,i,j] = linear_splines_var(precip[:,i,j]) 
            if error_calc:
                errors_unif[:,i,j] = calc_errors(precip[:,i,j], precip_unif[:,i,j])
                errors_var[:,i,j] = calc_errors(precip[:,i,j], precip_var[:,i,j])
                if errors_var[1,i,j] > errors_unif[1,i,j]:
                    precip_var[:,i,j] = precip_unif[:,i,j]
                    errors_var[:,i,j] = errors_unif[:,i,j]
        if include_obs:
            obs_crps[0,i,j] = obs_CRPS(obs[i,j], precip[:,i,j])            
            obs_crps[1,i,j] = obs_CRPS(obs[i,j], precip_unif[:,i,j])
            obs_crps[2,i,j] = obs_CRPS(obs[i,j], precip_var[:,i,j])    
    if save:
        precip_unif[level_idxs,:,:].dump('results/' + date + '/precip_unif_')
        precip_var[level_idxs,:,:].dump('results/' + date + '/precip_var_')
        if error_calc:
            errors_unif.dump('results/' + date + '/errors_unif_')
            errors_var.dump('results/' + date + '/errors_var_')
        if include_obs:
            obs_crps[0,:,:].dump('results/' + date + '/obs_crps_')        
            obs_crps[1,:,:].dump('results/' + date + '/obs_unif_crps_')
            obs_crps[2,:,:].dump('results/' + date + '/obs_var_crps_')                        

time_end = time.time()
print(f'Code completed in {(time_end - time_start) / 60 / 60} hours.')

# precip_unif = np.load('results/precip_unif_' + month, allow_pickle=True)
# precip_var = np.load('results/precip_var_' + month, allow_pickle=True)
# errors_unif = np.load('results/errors_unif_' + month, allow_pickle=True)
# errors_var = np.load('results/errors_var_' + month, allow_pickle=True) 
