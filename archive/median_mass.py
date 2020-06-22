# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:29:49 2018

@author: dori
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gammaincinv

def gammapsd(N0,LAM,mu,D):
    return N0*D**mu*np.exp(-LAM*D)

def cumgamma(N0,LAM,mu,D):
    return N0/(LAM**(mu+1))*gammainc(mu+1,LAM*D)

def cumgammainv(N0,LAM,mu,cum):
    return gammaincinv(mu+1, LAM**(mu+1.0)*cum/N0)/LAM

N0 = 1000
LAM = 1.0/3.0
mu = 2.0

Dmax = 50

D = np.linspace(0.01,Dmax,2000)
#plt.plot(D,gammapsd(N0,LAM,mu,D))
#plt.plot(D,cumgamma(N0,LAM,mu,D))


LAMs = 1.0/(np.linspace(0.2,3.0,30))
mus = 0.0+np.linspace(1.2,4.7,5)
plt.figure()
for m in mus:
    tots = cumgamma(N0,LAMs,m,1000)
    Dms = cumgammainv(N0,LAMs,m,0.5*tots)
    rhos = cumgamma(N0,LAMs,m,1000)/cumgamma(N0,LAMs,m+1,1000)
    #plt.loglog(LAMs,Dms)
    #plt.plot(LAMs,rhos)
    plt.semilogx(rhos,Dms)
        