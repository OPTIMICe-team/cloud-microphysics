import numpy as np
import pandas as pd

from pytmatrix import tmatrix, tmatrix_aux, radar, refractive

from microphysical_function	import Dice, Dcloud, Drain, gammaPSD


waves = {'X': tmatrix_aux.wl_X, 'Ka': tmatrix_aux.wl_Ka, 'W': tmatrix_aux.wl_W}


######################################################################################
# Generic Tmatrix function for single spheroid backscattering
######################################################################################
def tm_reflectivity(size, wl, n, ar=1.0): # Takes size and wl in meters
        scatt = tmatrix.Scatterer(radius=0.5e3*size, # conversion to millimeter radius
                                  radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM,
                                  wavelength=wl*1.0e3, # conversion to millimeters
                                  m=n,
                                  axis_ratio=1.0/ar)
        scatt.set_geometry(tmatrix_aux.geom_vert_back)
        return radar.radar_xsect(scatt) # mm**2 ... need just to integrate and multiply by Rayleigh factor

######################################################################################
# Liquid scattering routines - use Tmatrix
######################################################################################
scat_cloud = pd.DataFrame(index=Dcloud, columns=[w for w in waves.keys()])
scat_rain = pd.DataFrame(index=Drain, columns=[w for w in waves.keys()])
for w in waves.keys():
	n_water = refractive.m_w_0C[waves[w]]
	for i, d in enumerate(Dcloud):
		scat_cloud[i, w] = tm_reflectivity(d, 1.0e-3*waves[w], n_water)
	for i, d in enumerate(Drain):
		scat_rain[i, w] = tm_reflectivity(d, 1.0e-3*waves[w], n_water)


def calc_liquid_Z(N0, mu, lam, D, z):
	N = gammaPSD(N0, mu, lam)
    return (N(D)*z*np.gradient(D)).sum()
vector_liquid_Z = np.vectorize(calc_liquid_Z, excluded=['D', 'z'], otypes=[np.float])