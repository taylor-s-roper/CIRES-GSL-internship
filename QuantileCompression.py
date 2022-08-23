import numpy as np
import pygrib
from scipy import optimize 

class QuantileCompression:

    def __init__(self, zero_inflated=False, shift=0):

        self.zero_inflated = zero_inflated
        self.shift = shift
    
    def read_data(self, data=None, lat=None, long=None, file_path=None, msg_start=1, mask_data=False):
        '''
        Define data or data in grib file located at file_path. If mask_data is True, 
        data is converted into a masked array where cdf data that is not monotonic is masked.
        '''
        
        if data is None:
            ds_grb = pygrib.open(file_path)
            
            self.lat, self.long = ds_grb.message(msg_start).data()[1:]

            data = np.zeros((99,)+self.lat.shape)
            for i in range(99):
                data[i,:,:] = ds_grb.message(i + msg_start).data()[0]
            self.data = data[self.shift:]
            ds_grb.close()
            
        else:
            self.lat = lat
            self.long = long
            self.data = data[self.shift:]
            
        if mask_data:
            mask = np.zeros(self.data.shape)
            for i in range(self.lat.shape[0]):
                for j in range(self.lat.shape[1]):
                    issue = False
                    for level in range(self.data.shape[0]-1):
                        if self.data[level,i,j] > self.data[level+1,i,j]:
                            if not issue:
                                issue = True
                                mask[level+1:,i,j] = np.ones(mask.shape[0]-level-1)
            self.data = np.ma.masked_array(self.data, mask)
            self.mask = mask

        self.data_min = self.data[0,:,:].min()
        self.data_max = self.data[-1,:,:].max()
        self.mask_data = mask_data
    
    def fit_unif(self, data, num_knots=10, idx_start=None):  

        '''
        Calculates piecewise linear splines for quantile data using specified number of knots uniformly spaced and 
        returning 1-dimensional array consisting of [0,idx_start,knots[1:]] (knots[1:] is filled with 0s at 
        beginning in case there are not enough nonzero data points to use num_knots knots) where 0 is meant to 
        flag the output as comming from fit_unif.

        data: 1-dimensional array of shape = 99-shift containing cdf of variable at a grid point
        num_knots: int specifying number of knots/nodes to use
        idx_start: None or int specifying where zero_inflated data has last zero; it's passed in self.fit_var
        ''' 
        
        # returning zeros if cdf is all zero
        if data[-1] == 0:
            return np.zeros(num_knots+1)

        if idx_start is None:
            if self.zero_inflated: 
                # calculating where cdf starts being nonzero 
                idx_start = max(np.where(data > 0)[0].min() - 1, 0)
                data = data[idx_start:]
            else:
                idx_start = 0
        
        M = 99 - self.shift
        if M - idx_start < num_knots:
            knots = np.unique(np.linspace(idx_start+1, M, M-idx_start+1, dtype=int))
        else:
            knots = np.unique(np.linspace(idx_start+1, M, num_knots, dtype=int))  

        if self.zero_inflated:
            return np.hstack([0, idx_start, np.zeros(10-knots.shape[0]), data[knots[1:]-idx_start-1]])
        else:
            return np.hstack([0, data[knots-1]])

    def fit_var_func(self, x, num_knots, *params):

        '''
        Function to be used in scipy.optimize.curve_fit in linear_splines_var function.
        '''

        knot_vals = list(params[0][0:num_knots])
        knots = list(params[0][num_knots:])
        return np.interp(x, knots, knot_vals)

    def fit_var(self, data, num_knots=5, constrained=False):

        '''
        Calculates piecewise linear splines for quantile data using specified number of knots variably spaced and 
        returning 1-dimensional array consisting of [1,node_quantile,node_data] where 1 is meant to flag the output as comming from fit_var.

        data: 1-dimensional array of shape = 99-shift containing cdf of variable at a grid point
        num_knots: int specifying number of knots/nodes to use; if optimization fails, fit_unif is returned instead with the number of knots doubled
        '''
        
        # checking if cdf is all zero
        if data[-1] == 0:
            return np.zeros(2*num_knots + 1)
        
        M = 99 - self.shift
        if self.zero_inflated:
            # calculating where cdf starts being nonzero (all zero cdf's should not be inputted)
            idx_start = max(np.where(data > 0)[0].min() - 1, 0)
            data = data[idx_start:]
            if M-idx_start < num_knots*2:
                return self.fit_unif(data, num_knots=2*num_knots, idx_start=idx_start)
        else:
            idx_start = 0
            
        # setting up intial value of parameters
        levels = np.linspace(1,99,99)[self.shift:]
        p0 = np.linspace(0, M-idx_start-1, num_knots)
        p0 = np.hstack([np.interp(p0+idx_start+1+self.shift, levels[idx_start:], data), p0])

        # try to fit parameters with RuntimeError exception that returns fit_unif
        # that uses uniformly space knots
        try:
            if constrained:
                bounds = np.hstack([data[0] * np.ones(num_knots), np.zeros(num_knots)])
                bounds = (bounds,) + (np.hstack([data[-1] * np.ones(num_knots), 100 * np.ones(num_knots)]),)
                fit, _ = optimize.curve_fit(lambda x, *params : self.fit_var_func(x, num_knots, params), xdata=levels[idx_start:], ydata=data, p0=p0, bounds=bounds)
            else:
                fit, _ = optimize.curve_fit(lambda x, *params : self.fit_var_func(x, num_knots, params), xdata=levels[idx_start:], ydata=data, p0=p0)
            sort_idx = np.argsort(fit[num_knots:])
            fit = np.hstack([fit[:num_knots][sort_idx], fit[num_knots:][sort_idx]])
            for k in range(num_knots-1):
                if fit[:num_knots][k+1] < fit[:num_knots][k]:
                    return self.fit_unif(data, num_knots=2*num_knots, idx_start=idx_start)
            if self.zero_inflated:
                return np.hstack([1, 0, fit[1:num_knots], idx_start, fit[num_knots+1:]])
            else:
                return np.hstack([1, fit])
        except RuntimeError:
            return self.fit_unif(data, num_knots=2*num_knots, idx_start=idx_start)

    def calc_error(self, orig, approx, error_type='mae', N=int(1e3)):
        '''
        Calculating either maximum absolute error, mean absolute error, or mean square error.

        orig: original forecast data at grid point
        approx: approximate forecast data at grid point
        error_type: string of either 'max' for maximum absolute error, 'mae' for mean absolute error, or 'mse' for mean square error
        N: int corresponding to number of samples for error estimates
        '''

        if orig[-1] == 0 and approx[-1] == 0:
            return 0
        else:
            xs = np.linspace(self.data_min, self.data_max, N)
            qs = np.linspace(1,99,99)[self.shift:]/100
            differences = np.abs(np.interp(xs[1:-1], orig[1:-1], qs[1:-1], left=0.0, right=1.0) - np.interp(xs[1:-1], approx[1:-1], qs[1:-1], left=0.0, right=1.0))
            if error_type == 'max':
                return differences.max()
            elif error_type == 'mae':
                return np.mean(differences)
            elif error_type == 'mse':
                return np.mean(differences**2)
            else:
                print('Error types are max, mae, or mse. Assuming mae.')
                return np.mean(differences)

    def compress_data(self, compress_type='unif', num_data_pts=10, tol=1e-3):
        '''
        Compressing gridded quantile data and returns gridded data containing fit outputs

        data: gridded qunatile data to be compressed
        compress_type: string of either 'unif' for uniformly spaced nodes, 'var' for 
        unbounded variably spaced nodes, 'var_const' for bounded variably spaced nodes, or 'adapt' 
        for a mix of the previous three based on minimum error
        
        num_data_pts: int corresponding to number of data points used
        tol: error tolerance for type 'adapt'
        '''

        if self.mask_data:
            idx = np.where(self.mask[-1,:,:] == 0)
        else:
            idx = np.where(self.data[-1,:,:] != 0)
        compressed_data = np.zeros((num_data_pts+1,) + self.lat.shape)
        for n in range(idx[0].shape[0]):
            i = idx[0][n]
            j = idx[1][n]
            if compress_type == 'unif':
                compressed_data[:,i,j] = self.fit_unif(self.data[:,i,j], num_knots=num_data_pts)
            elif compress_type == 'var':
                compressed_data[:,i,j] = self.fit_var(self.data[:,i,j], num_knots=int(num_data_pts/2))
            elif compress_type == 'var_const':
                compressed_data[:,i,j] = self.fit_var(self.data[:,i,j], num_knots=int(num_data_pts/2), constrained=True)
            elif compress_type == 'adapt':
                fit_unif = self.fit_unif(self.data[:,i,j], num_knots=num_data_pts)
                unif = self.decompress_func(fit_unif)
                unif_error = self.calc_error(self.data[:,i,j], unif)
                if unif_error < tol:
                    compressed_data[:,i,j] = fit_unif
                else:
                    fit_var = self.fit_var(self.data[:,i,j], num_knots=int(num_data_pts/2))
                    var = self.decompress_func(fit_var)
                    var_error = self.calc_error(self.data[:,i,j], var)
                    if var_error < tol:
                        compressed_data[:,i,j] = fit_var
                    else:
                        fit_var_const = self.fit_var(self.data[:,i,j], num_knots=int(num_data_pts/2), constrained=True)
                        var_const = self.decompress_func(fit_var_const)
                        var_const_error = self.calc_error(self.data[:,i,j], var_const)
                        if var_const_error < tol:
                            compressed_data[:,i,j] = fit_var_const
                        else:
                            errors = np.hstack([unif_error, var_error, var_const_error])
                            fits = [fit_unif, fit_var, fit_var_const]
                            compressed_data[:,i,j] = fits[np.where(errors == errors.min())[0][0]]

        return compressed_data
                
    def decompress_func(self, fit):
        '''
        Decompression function that takes fit parameters at a grid point and returns approximate cdf

        fit: array of 0 or 1 flag plus number of data points used
        '''

        qs = np.linspace(1,99,99)[self.shift:]/100
        num_data_pts = fit.shape[0] - 1
        M = 99 - self.shift
        if self.zero_inflated:
            if fit[0] == 0:
                idx_start = int(fit[1])
                if M - idx_start < num_data_pts:
                    knots = (np.unique(np.linspace(idx_start+1, M, M-idx_start+1, dtype=int)) + self.shift)/100
                else:
                    knots = (np.unique(np.linspace(idx_start+1, M, num_data_pts, dtype=int)) + self.shift)/100
                return np.interp(qs, np.hstack([(idx_start+1+self.shift)/100, knots[1:]]), np.hstack([0, fit[-knots.shape[0]+1:]]))
            else:
                fit = fit[1:]
                return np.interp(qs, (fit[int(num_data_pts/2):] + self.shift)/100, fit[:int(num_data_pts/2)])
        else:
            if fit[0] == 0:
                knots = (np.unique(np.linspace(1, M, num_data_pts, dtype=int)) + self.shift)/100
                return np.interp(qs, knots, fit[-num_data_pts:])
            else:
                fit = fit[1:]
                return np.interp(qs, (fit[int(num_data_pts/2):] + self.shift)/100, fit[:int(num_data_pts/2)])

    def decompress_data(self, fit_grid):
        '''
        Decompresses gridded data with fit parameters.

        fit_grid: 3-dimensional array of gridded data with fit parameters as first dimension
        '''

        decompressed_data = np.zeros((99-self.shift,) + fit_grid[0,:,:].shape)
        if self.mask_data:
            decompressed_data = np.ma.masked_array(decompressed_data, self.mask)
            idx = np.where(self.mask[-1,:,:] == 0)
        else:
            idx = np.where(decompressed_data[-1,:,:] == 0)
        for n in range(idx[0].shape[0]):
            i = idx[0][n]
            j = idx[1][n]
            decompressed_data[:,i,j] = self.decompress_func(fit_grid[:,i,j])

        return decompressed_data
    
    def error_summaries(self, decompressed_data):
        '''
        Calculates and prints error summaries

        decompressed_data: array of approximate data with same shape as original data
        '''
        
        errors = np.zeros((3,) + self.lat.shape)
        if self.mask_data:
            idx = np.where(self.mask[-1,:,:] == 0)
        else:
            idx = np.where(self.data[-1,:,:] != 0)

        for n in range(idx[0].shape[0]):
            i = idx[0][n]
            j = idx[1][n]
            errors[0,i,j] = self.calc_error(self.data[:,i,j], decompressed_data[:,i,j], error_type='max')
            errors[1,i,j] = self.calc_error(self.data[:,i,j], decompressed_data[:,i,j], error_type='mae')
            errors[2,i,j] = self.calc_error(self.data[:,i,j], decompressed_data[:,i,j], error_type='mse')

        if self.zero_inflated:
            avg_errors = np.zeros(3)
            tot = 0
            for n in range(idx[0].shape[0]):
                i = idx[0][n]
                j = idx[1][n]
                if self.data[-1,i,j] != 0:
                    tot += 1
                    avg_errors += errors[:,i,j]
            avg_errors = avg_errors/tot
        else:
            avg_errors = [np.mean(errors[0,:,:]), np.mean(errors[1,:,:]), np.mean(errors[2,:,:])]
        
        print(f'max absolute error = {errors[0,:,:].max()}')
        print(f'max mean absolute error over grid = {errors[1,:,:].max()}')
        print(f'max mean squared error over grid = {errors[2,:,:].max()}')
        print(f'average max absolute error over grid = {avg_errors[0]}')
        print(f'average mean absolute error over grid = {avg_errors[1]}')
        print(f'average mean squared error over grid = {avg_errors[2]}')