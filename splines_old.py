# read in grib file
date = '20220324'
issuance_time = '12'
lead_time = '018'
obs_time = '0600'
fn_grb = 'data/' + date + '/blend' + date + '.t' + issuance_time + 'z.qmd.f' + lead_time + '.co.grib2'
ds_grb = pygrib.open(fn_grb)

# latitude and longitude grid values
lat, long = ds_grb.message(2).data()[1:]

# extracting precipitation data
msg_start = 2
data = np.zeros(shape=(99,)+lat.shape)
for i in range(99):
    tmp = ds_grb.message(i+msg_start).data()[0]
    data[i,:,:] = tmp
data_scaled = data[:,:1597-220,200:]
ds_grb.close()

# extracting temperature data
# ds_grb = pygrib.open(fn_grb)
# msg_start = 213
# data = np.zeros(shape=(99,)+lat.shape)
# for i in range(99):
#     data[i,:,:] = ds_grb.message(i+msg_start).data()[0]
# ds_grb.close()
# data = data[4:] #removing 1%-4% quantiles for temp data

# lat = lat[:1597-220,200:]
# long = long[:1597-220,200:]
# data = data[:,:1597-220,200:]

# extracting observed precipitation data
time_ = '01'
# time_ = '13' 
obs_date = '20220325'
fn_grb = 'data/' + date + '/rtma2p5.' + obs_date + time_ + '.pcp.184.grb2'
ds_grb = pygrib.open(fn_grb)
lat_scaled, long_scaled = ds_grb.message(1).data()[1:]
tmp = ds_grb.message(1).data()[0]
obs = tmp
ds_grb.close()
# for time_ in ['03', '04', '05', '06']:
for time_ in ['02', '03', '04', '05', '06']:
# for time_ in ['14', '15', '16', '17', '18']:
    fn_grb = 'data/' + date + '/rtma2p5.' + obs_date + time_ + '.pcp.184.grb2'
    ds_grb = pygrib.open(fn_grb)
    tmp = ds_grb.message(1).data()[0]
    obs += tmp
    ds_grb.close()

# extracting observed data
# fn_grb = 'data/' + date + '/urma2p5.2022062612.pcp_06h.wexp.grb2' #for 20220626
# fn_grb = 'data/' + date + '/rtma2p5_ru.t' + obs_time + 'z.2dvaranl_ndfd.grb2' # for temp data
# ds_grb = pygrib.open(fn_grb)
# obs = ds_grb.message(3).data()[0] # message=3 for temp data
# ds_grb.close()

# masking precip at grid points that are not monotonic - not needed for temperature data
mask = np.zeros(data.shape)
for i in range(lat.shape[0]):
    for j in range(lat.shape[1]):
        issue = False
        for level in range(data.shape[0]-1):
            if data[level,i,j] > data[level+1,i,j]:
                if not issue:
                    issue = True
                    mask[level+1:,i,j] = np.ones(mask.shape[0]-level-1)
mask_scaled = mask[:,:1597-220,200:]
data = np.ma.masked_array(data, mask)
data_scaled = np.ma.masked_array(data_scaled, mask_scaled)
obs = np.ma.masked_array(obs, mask_scaled[-1,:,:])

# initializing variables
levels = np.linspace(1,99,99)
# levels = levels[4:] #changed for temp data
qs = levels / 100
N = int(1e3) # number of samples for error estimates
data_max = max(obs.max(), data[-1,:,:].max()) # extreme data values for error estimates
data_min = min(obs.min(), data[0,:,:].min()) 
# data_max = data[-1,:,:].max() # if no observed data
# data_min = data[0,:,:].min() 
xs = np.linspace(data_min, data_max, N) # data values to integrate over for error estimates
M = data.shape[0]
bounds = np.hstack([data_min * np.ones(5), np.zeros(5)])
bounds = (bounds,) + (np.hstack([data_max * np.ones(5), 100 * np.ones(5)]),)

# linear spline functions
def linear_splines_unif(data, num_knots=10, zero_inflated=False, idx_start=None, shift=0):   
    '''
    Calculates piecewise linear splines for quantile data using specified number of 
    knots uniformly spaced and returning interpolated approximation at every 
    quantile level.
    ''' 
    
    # checking if cdf is all zero
    if data[-1] == 0:
        return np.zeros(M)

    if idx_start is None:
        if zero_inflated: # idx_start may be calculated in linear_splines_var code which automatically means zero_inflated=True
            # calculating where cdf starts being nonzero (all zero cdf's should not be inputted)
            idx_start = max(np.where(data > 0)[0].min() - 1, 0)
            data = data[idx_start:]
        else:
            idx_start = 0
    
    if M-idx_start < num_knots:
        knots = np.unique(np.linspace(idx_start+1, M, M-idx_start+1, dtype=int))
    else:
        knots = np.unique(np.linspace(idx_start+1, M, num_knots, dtype=int))

    if zero_inflated:
        return np.interp(levels, knots+shift, data[knots-idx_start-1], left=0.0)
    else:
        return np.interp(levels, knots+shift, data[knots-1])

def linear_splines(x, num_knots, shift, *params):
    '''
    Function to be used in scipy.optimize.curve_fit in linear_splines_var function.
    '''

    knot_vals = np.array(params[0][0:num_knots])
    knots = np.array(params[0][num_knots:])
    return np.interp(x, knots+1+shift, knot_vals)

def linear_splines_var(data, num_knots=5, zero_inflated=False, method='lm', shift=0):
    '''
    Calculates piecewise linear splines for quantile data using specified number of
    knots with optimized placement and returning interpolated approximation at every
    quantile level with level_width.
    '''
    
    # checking if cdf is all zero
    if data[-1] == 0:
        return np.zeros(M)
    
    if zero_inflated:
        # calculating where cdf starts being nonzero (all zero cdf's should not be inputted)
        idx_start = max(np.where(data > 0)[0].min() - 1, 0)
        data = data[idx_start:]
        if M-idx_start < num_knots*2:
            return linear_splines_unif(data, num_knots=2*num_knots, idx_start=idx_start, zero_inflated=zero_inflated, shift=shift)
    else:
        idx_start = 0
        
    # setting up intial value of parameters
    #p0 = np.linspace(idx_start+1,M,num_knots)
    p0 = np.linspace(0, M-idx_start-1, num_knots)
    p0 = np.hstack([np.interp(p0+idx_start+1+shift, levels[idx_start:], data), p0])

    # try to fit parameters with RuntimeError exception that returns linear_splines_unif that uses uniformly space knots
    try:
        if method == 'lm':
            fit, _ = optimize.curve_fit(f=lambda x, *params : linear_splines(x, num_knots, shift, params), xdata=levels[idx_start:], ydata=data, p0=p0, method=method)
        else:
            fit, _ = optimize.curve_fit(f=lambda x, *params : linear_splines(x, num_knots, shift, params), xdata=levels[idx_start:], ydata=data, p0=p0, bounds=bounds, method=method)
        sort_idx = np.argsort(fit[num_knots:])
        fit = np.hstack([fit[:num_knots][sort_idx], fit[num_knots:][sort_idx]])
        # if fit[:num_knots][0] < data_min or fit[:num_knots][-1] > data_max:
        #     return linear_splines_unif(data, num_knots=2*num_knots, idx_start=idx_start, zero_inflated=zero_inflated)
        for k in range(num_knots-1):
            if fit[:num_knots][k+1] < fit[:num_knots][k]:
                return linear_splines_unif(data, num_knots=2*num_knots, idx_start=idx_start, zero_inflated=zero_inflated, shift=shift)
        return np.hstack([np.zeros(idx_start), np.interp(levels[idx_start:], fit[num_knots:]+1+shift, fit[:num_knots])])
    except RuntimeError:
        return linear_splines_unif(data, num_knots=2*num_knots, idx_start=idx_start, zero_inflated=zero_inflated, shift=shift)

# error functions
def calc_errors(orig, approx):
    '''
    Calculating the Kolmogorov-Smirnov (KS) statistic, average L_1 norm of difference, and Continuous Ranked Probability Score (CRPS)
    '''
    if orig[-1] == 0 and approx[-1] == 0:
        return np.zeros(3)
    else:
        differences = np.abs(np.interp(xs[1:-1], orig[1:-1], qs[1:-1], left=0.0, right=1.0) - np.interp(xs[1:-1], approx[1:-1], qs[1:-1], left=0.0, right=1.0))
        return [differences.max(), np.mean(differences), np.mean(differences**2)] # [KS, mean L_1, CRPS]

def obs_CRPS(obs, frcst):
    '''
    Calculating CRPS between observed data and forecasted data
    '''
    obs_cdf = np.zeros(N-2)
    obs_nonzero = np.where(xs[1:-1] >= obs)[0]
    obs_cdf[obs_nonzero] = np.ones(obs_nonzero.shape[0])
    return np.mean((np.interp(xs[1:-1], frcst[1:-1], qs[1:-1], left=0.0, right=1.0) - obs_cdf)**2)

# initializing variables
# unif = np.ma.masked_array(np.zeros(data.shape), mask)
var = np.ma.masked_array(np.zeros(data.shape), mask)
# errors_unif = np.ma.masked_array(np.zeros((3,) + lat.shape), [mask[-1,:,:],mask[-1,:,:],mask[-1,:,:]])
errors_var = np.ma.masked_array(np.zeros((3,) + lat.shape), [mask[-1,:,:],mask[-1,:,:],mask[-1,:,:]])  
# obs_crps = np.ma.masked_array(np.zeros((3,) + lat_scaled.shape), [mask_scaled[-1,:,:],mask_scaled[-1,:,:],mask_scaled[-1,:,:]])
obs_crps = np.ma.masked_array(np.zeros(lat_scaled.shape), mask_scaled[-1,:,:])
# unif = np.zeros(data.shape)
# var = np.zeros(data.shape)
# errors_unif = np.zeros((3,) + lat.shape)
# errors_var = np.zeros((3,) + lat.shape)
# obs_crps = np.zeros((3,) + lat.shape)
# obs_crps = np.zeros(lat.shape)

idx = np.where(mask[-1,:,:] == 0)
# idx = np.where(data[-1,:,:] != 0)
level_idxs = [4, 24, 50, 74, 94] # saving data, unif, and var outputs at 5th, 25th, 50th, 75th, and 95th quantiles
# level_idxs = [0, 20, 46, 70, 90]

print(f'{datetime.now().strftime("%H:%M:%S")}: Main program block starting.')

for n in range(idx[0].shape[0]):
    i = idx[0][n]
    j = idx[1][n]

    if data[-1,i,j] != 0:
        # uniformly spaced nodes
        # unif[:,i,j] = linear_splines_unif(data[:,i,j], zero_inflated=True)
        # errors_unif[:,i,j] = calc_errors(data[:,i,j], unif[:,i,j])

        #variably spaced nodes
        var[:,i,j] = linear_splines_var(data[:,i,j], zero_inflated=True)      
        errors_var[:,i,j] = calc_errors(data[:,i,j], var[:,i,j]) 

        # replacing var with unif if L_1 error of unif is smaller than of var
        #if errors_var[1,i,j] > errors_unif[1,i,j]:
        #    var[:,i,j] = unif[:,i,j]
        #    errors_var[:,i,j] = errors_unif[:,i,j]

    # end of if block
    
    if i < 1597-220 and j >= 200:
        # obs_crps[0,i,j-200] = obs_CRPS(obs[i,j-200], data[:,i,j])
        # obs_crps[1,i,j-200] = obs_CRPS(obs[i,j-200], unif[:,i,j])
        # obs_crps[2,i,j-200] = obs_CRPS(obs[i,j-200], var[:,i,j])
        obs_crps[i,j-200] = obs_CRPS(obs[i,j-200], var[:,i,j])

    # obs_crps[0,i,j] = obs_CRPS(obs[i,j], data[:,i,j])            
    # obs_crps[1,i,j] = obs_CRPS(obs[i,j], unif[:,i,j])
    # obs_crps[2,i,j] = obs_CRPS(obs[i,j], var[:,i,j])  
    # obs_crps[i,j] = obs_CRPS(obs[i,j], var[:,i,j])

    if n == int(idx[0].shape[0]/4):
        print(f'{datetime.now().strftime("%H:%M:%S")}: Program is ~25% complete.')
    elif n == int(idx[0].shape[0]/2):
        print(f'{datetime.now().strftime("%H:%M:%S")}: Program is ~50% complete.')
    elif n == int(3*idx[0].shape[0]/4):
        print(f'{datetime.now().strftime("%H:%M:%S")}: Program is ~75% complete.')


# unif[level_idxs,:,:].dump('results/' + date + '/precip_unif')
var[level_idxs,:,:].dump('results/' + date + '/precip_var_')
# errors_unif.dump('results/' + date + '/precip_errors_unif')
errors_var.dump('results/' + date + '/precip_errors_var_')
obs_crps.dump('results/' + date + '/precip_obs_var_crps_')
# unif[level_idxs,:,:].dump('results/' + date + '/temp_unif')
# var[level_idxs,:,:].dump('results/' + date + '/temp_var_')
# errors_unif.dump('results/' + date + '/temp_errors_unif')
# errors_var.dump('results/' + date + '/temp_errors_var_')
# obs_crps.dump('results/' + date + '/temp_obs_var_crps_')        

time_end = time.time()

print(f'{datetime.now().strftime("%H:%M:%S")}: Code complete!')
print(f'{datetime.now().strftime("%H:%M:%S")}: Time elapsed = {(time_end - time_start) / 60} minutes.')