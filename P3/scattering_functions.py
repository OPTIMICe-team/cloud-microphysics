import numpy as np
from pytmatrix import tmatrix
from pytmatrix import tmatrix_aux
from pytmatrix import radar


ice_density = 900.0 # limited ice density from P3


def spheroid_volume(d, ar):
	ar*np.pi*d**3/6.0

	
def solid_spheroid_mass(d, ar):
	ice_density*spheroid_volume(d, ar)
	

waves = {'X': tmatrix_aux.wl_X, 'Ka': tmatrix_aux.wl_Ka, 'W': tmatrix_aux.wl_W}


def tm_reflectivity(size, wl, n, ar=1.0):
        scatt = tmatrix.Scatterer(radius=0.5e3*size, # conversion to millimeter radius
                                  radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM,
                                  wavelength=wl*1.0e3, # conversion to millimeters
                                  m=n,
                                  axis_ratio=1.0/ar)
        scatt.set_geometry(tmatrix_aux.geom_vert_back)
        return radar.radar_xsect(scatt) # mm**2 ... need to integrate and multiply by Rayleigh factor