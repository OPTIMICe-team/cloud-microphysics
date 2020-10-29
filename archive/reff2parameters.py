#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:10:46 2020

@author: dori
"""

import numpy as np
from scipy.special import gamma


def transformPSD(a=np.nan, b=np.nan, 
                 mu=np.nan, gam=np.nan, 
                 xmin=np.nan, xmax=np.nan,
                 N0=np.nan, lam=np.nan):
    """
    This is a converter.
    It converts parameters of a modified-gamma mass distribution written like
    
    (1) Nm(m) = N0 m**mu * np.exp(-lam m**gam)
    
    to the corresponding N0, mu, lam, gam parameters of the size distribution
    
    (2) ND(D) = N0 D**mu * np.exp(-lam D**gam)
    
    given the power law relation between size and mass
    
    (3) D(m) = a*m**b
    
    and since we are here, also the converted mass-size relation
    
    (4) m(D) = am* D**bm
    
    NOTE: I have reused the terms N0, mu, lam, gam in equation (1) and (2)
    because it is easier to follow like that, but the whole point of this
    function is that they must be converted.
    I should have used N0m mum lamm and gamm like I did in eq (4), but I am
    lazy
    
    Whatever the function cannot calculate because of insufficient arguments
    is left to np.nan
    
    """
    bm  = 1.0/b
    am  = 1.0/a**bm
    N0m  = am**(mu+1.0)*bm*N0
    mum = mu*bm + bm - 1.0
    lamm = lam*am**gam
    gamm = bm*gam
    Dmin= a*xmin**b
    Dmax= a*xmax**b
    
    return am, bm, N0m, mum, lamm, gamm, Dmin, Dmax

# Some arguments I took from CR-SIM manual
cloud = {'a':0.124, 'b':1.0/3.0, 'mu':1.0, 'gam':1.0}
rain  = {'a':0.124, 'b':1.0/3.0, 'mu':0.0, 'gam':1.0/3.0}
ice   = {'a':0.835, 'b':0.39,    'mu':0.0, 'gam':1.0/3.0}
snow  = {'a':5.13,  'b':0.5,     'mu':0.0, 'gam':0.5}
graup = {'a':0.142, 'b':0.314,   'mu':1.0, 'gam':1.0/3.0}
hail  = {'a':0.1366,'b':1.0/3.0, 'mu':1.0, 'gam':1.0/3.0}

# Let's say I want to convert the parameters for snow
am, bm, N0m, mum, lamm, gamm, Dmin, Dmax = transformPSD(**snow)
# N0m, lamm, Dmin, Dmax will remain np.nan because I do not know lam, N0 xmin and xmax ...

# Now let's constrain N0 and lam using reff and q
def reff2lamm(reff, mum, gamm):
    """
    The effective radius does not depend on N0, so it constrain directly lamm
    """
    return (gamma((mum+4.0)/gamm)/(2.0*reff*gamma((mum+3.0)/gamm)))**gamm

def q2N0m(q, gamm, lamm, mum, am, bm):
    """
    N0 is just a scaling factor. If we have everything else we only need to
    match mixing ratio q
    """
    return q*gamm*lamm**((mum+bm+1.0)/gamm)/(am*gamma((mum+bm+1.0)/gamm))

# Remember everything is assumed to be in SI units mks, that comes from the 
# definition of the hydrometeor properties. Thus reff is expected to be meters
# and q should be kg/kg

# As an example for snow with reff 5 mm and q = 1e-2 (I know is a lot, but 5 mm is quite huge as well...)
lamm = reff2lamm(0.005, mum, gamm)
N0m = q2N0m(1e-2, gamm, lamm, mum, am, bm)


# Now you have everything ou need to calculate what you want. But just for fun:
def momentPSD(N0, mum, lamm, gamm, k):
    """
    This function calculates the generic moment k of the PSD defined as
    modified gamma
    """
    return N0m*gamma((mum+k+1.0)/gamm)/(gamm*lamm**((mum+k+1.0)/gamm))

# Total number concentration. Should be something around some thousand
Nt = momentPSD(N0m, mum, lamm, gamm, 0)

# mixing ratio. Should be exactly 1e-2
q = am*momentPSD(N0m, mum, lamm, gamm, bm)

# effective radius. Should be exactly 5 millimeters
reff = 0.5*momentPSD(N0m, mum, lamm, gamm, 3)/momentPSD(N0m, mum, lamm, gamm, 2)





