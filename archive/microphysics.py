# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:28:34 2017

@author: dori
"""

import numpy as np
from scipy.special import gamma

def set_hydro_params(hydro):
    if hydro=='snowSBB':
        return 8.294, 0.125, 0.0, 0.5
    if hydro=='snow_cosmo5':
        return 8.8, 0.15, 0.0, 0.5
    if hydro=='ice_cosmo5':
        return 27.7, 0.21579, 0.0, 1.0/3.0
    else:
        print('Unknown hydrometeor model')
        return np.nan,np.nan,np.nan,np.nan

def SEDIVEL_ICON(q,N,moment,hydro):
    a_vel,b_vel,nu,mu = set_hydro_params(hydro)
    first_term = a_vel*gamma((moment+nu+b_vel+1.0)/mu)/gamma((moment+nu+1.0)/mu)
    second_term = (gamma((nu+1.0)/mu)/gamma((nu+2.0)/mu))**b_vel
#    try:
#        mean_mass = q/N
#    except ZeroDivisionError:
#        mean_mass = np.nan
    if q < 0.000001:
        return np.nan
    elif N < 0.1:
        return np.nan
    mean_mass = q/N
    return first_term * second_term * mean_mass**b_vel

vSEDIVEL_ICON = np.vectorize(SEDIVEL_ICON)