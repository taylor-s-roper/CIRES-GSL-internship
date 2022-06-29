from dataclasses import dataclass
import numpy as np
import pygrib
from scipy import interpolate, optimize, stats

class Compression:
    '''
    Class for gridded data compression using either linear splines
    with uniform or variable node placement or fitted gamma distribution.
    Gridded data is expected to contain qunatile information for 
    every percentage from 1 to 99, i.e. 1%, 2%, 3%, ....
    '''

    def __init__(self, data, method='unif')
    '''
    data: array like containing gridded data
    method: string specifying approximation method used. Options are
        'unif', 'var', and 'gamma'.
    '''

    if method == 'unif':

    elif method == 'var':

    elif method == 'gamma':

    else:
        print(f'{method} is not a recognized method. Try unif, var, or gamma.')

    self.data = data

    def unif(self):
        