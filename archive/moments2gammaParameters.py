#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:31:48 2018

@author: dori
"""

from scipy import special
import numpy as np

dw = 1000.0 # kg/m3 density of water

def Nq2LambdaN0(N, q, mu=2.0, gamma=1.0):
  Lambda = (dw*np.pi*q*special.gamma((mu+4.0)/gamma)/(6.0*N*special.gamma((mu+1.0)/gamma)))**(gamma/3.0)
  N0 = N*gamma*Lambda**((mu+1.0)/gamma)/special.gamma((mu+1.0)/gamma)