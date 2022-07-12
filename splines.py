
approx_type = 'both' # 'unif', 'var', or 'both'
include_temp = False # does not approximate temperature data yet
error_calc = True
save = True
month = 'june'
include_obs = True
# months completed: jan, march, april, july

import numpy as np
import pygrib
from scipy import optimize

import time

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
    fn_grb = 'urma2p5.2022062612.pcp_06h.wexp.grb2'
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

# linear spline functions

def linear_splines_unif(data, num_knots=10, zero_inflated=True):   
    '''
    Calculates piecewise linear splines for quantile data using specified number of 
    knots uniformly spaced and returning interpolated approximation at every 
    quantile level.
    ''' 

    # checking if cdf is all zero
    if data[-1] == 0:
        return np.zeros(99)
    
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
    if data[-1] == 0:
        return np.zeros(99)

    # setting up intial value of parameters
    p_0 = np.linspace(0,98,num_knots).astype(int)
    p_0 = np.hstack([data[p_0], p_0])

    # try to fit parameters with RuntimeError exception that returns linear_splines_unif
    # that uses uniformly space knots
    try:
        fit, _ = optimize.curve_fit(lambda x, *params : linear_splines(x, num_knots, params), np.linspace(1,99,99), data, p_0)
        levels = range(1,100)
        return np.interp(levels, fit[num_knots:], fit[:num_knots])
    except RuntimeError:
        return linear_splines_unif(data, num_knots*2, zero_inflated)
    
qs = np.linspace(0.01, 0.99, 99)
N = int(1e3)

def calc_errors(orig, approx):
    xs = np.linspace(orig.min(), orig.max(), N)
    differences = np.abs(np.interp(xs[1:], orig, qs) - np.interp(xs[1:], approx, qs))
    differences_weighted = differences * (xs[1:] - xs[:-1])
    return [differences.max(), np.sum(differences_weighted), np.sum(differences * differences_weighted)]

def obs_CRPS(obs, approx):
    xs = np.linspace(approx.min(), approx.max(), N)
    obs_cdf = np.zeros(xs.shape[0])
    obs_nonzero = np.where(xs >= obs)[0]
    obs_cdf[obs_nonzero] = np.ones(obs_nonzero.shape[0])
    return np.sum((np.interp(xs[1:], approx, qs) - obs_cdf[1:])**2 * (xs[1:] - xs[:-1]))

precip_unif = np.ma.masked_array(np.zeros(precip.shape), mask)
precip_var = np.ma.masked_array(np.zeros(precip.shape), mask)
if error_calc:
    errors_unif = np.ma.masked_array(np.zeros((3,) + lat.shape), [mask[-1,:,:],mask[-1,:,:],mask[-1,:,:]])
    errors_var = np.ma.masked_array(np.zeros((3,) + lat.shape), [mask[-1,:,:],mask[-1,:,:],mask[-1,:,:]])    
if include_obs:
    obs_crps = np.ma.masked_array(np.zeros((2,)+lat.shape), [mask[-1,:,:],mask[-1,:,:]])
idx = np.where(mask[-1,:,:] == 0)
level_idxs = [4, 24, 50, 74, 94]

if approx_type == 'unif':
    for n in range(idx[0].shape[0]):
        i = idx[0][n]
        j = idx[1][n]
        if precip[-1,i,j] != 0:
            precip_unif[:,i,j] = linear_splines_unif(data=precip[:,i,j], num_knots=10, zero_inflated=True)  
            if error_calc:
                errors_unif[:,i,j] = calc_errors(precip[:,i,j], precip_unif[:,i,j])
        if include_obs:
            obs_crps[0,i,j] = obs_CRPS(obs[i,j], precip_unif[:,i,j])
    if save:
        precip_unif[level_idxs,:,:].dump('results/precip_unif_' + month)
        errors_unif.dump('results/errors_unif_' + month)
        if include_obs:
            obs_crps[0,:,:].dump('results/obs_unif_crps_' + month)

elif approx_type == 'var':
    for n in range(idx[0].shape[0]):
        i = idx[0][n]
        j = idx[1][n]
        if precip[-1,i,j] != 0:
            precip_var[:,i,j] = linear_splines_var(data=precip[:,i,j], num_knots=10, zero_inflated=True) 
            if error_calc:
                errors_var[:,i,j] = calc_errors(precip[:,i,j], precip_var[:,i,j])
        if include_obs:
            obs_crps[1,i,j] = obs_CRPS(obs[i,j], precip_var[:,i,j])    
    if save:
        precip_var[level_idxs,:,:].dump('results/precip_var_' + month)
        errors_var.dump('results/errors_var_' + month)
        if include_obs:
            obs_crps[1,:,:].dump('results/obs_var_crps_' + month)

elif approx_type == 'both':
    for n in range(idx[0].shape[0]):
        i = idx[0][n]
        j = idx[1][n]
        if precip[-1,i,j] != 0:
            precip_unif[:,i,j] = linear_splines_unif(data=precip[:,i,j], num_knots=10, zero_inflated=True)  
            precip_var[:,i,j] = linear_splines_var(data=precip[:,i,j], num_knots=10, zero_inflated=True) 
            if error_calc:
                errors_unif[:,i,j] = calc_errors(precip[:,i,j], precip_unif[:,i,j])
                errors_var[:,i,j] = calc_errors(precip[:,i,j], precip_var[:,i,j])
        if include_obs:
            obs_crps[0,i,j] = obs_CRPS(obs[i,j], precip_unif[:,i,j])
            obs_crps[1,i,j] = obs_CRPS(obs[i,j], precip_var[:,i,j])    
    if save:
        precip_unif[level_idxs,:,:].dump('results/precip_unif_' + month)
        errors_unif.dump('results/errors_unif_' + month)
        precip_var[level_idxs,:,:].dump('results/precip_var_' + month)
        errors_var.dump('results/errors_var_' + month)
        obs_crps.dump('results/obs_crps_' + month)

# precip_unif = np.load('results/precip_unif_' + month, allow_pickle=True)
# precip_var = np.load('results/precip_var_' + month, allow_pickle=True)
# errors_unif = np.load('results/errors_unif_' + month, allow_pickle=True)
# errors_var = np.load('results/errors_var_' + month, allow_pickle=True) 