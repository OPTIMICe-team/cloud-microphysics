# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:26:41 2018

@author: dori
"""

from scipy import special

def vsed2lam(vsed, av, bv, mu=1.0):
    return (av*special.gamma((bv+7.0)/mu)/(special.gamma(7.0/mu)*vsed))**(mu/bv)

def Z2N(z,lam):
    return 1.0e-18 * 10.0**(z*0.1) * lam**7.0 / special.gamma(7)