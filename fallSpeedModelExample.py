 # The speed of light in the vacuum m/s. This is the TRUE value!!
_c = 2.99792458e8
 # a value for the density of solid ice Ih kg/m**3
_ice_density = 917.0
# a value for gravity acceleration m/s**2
_g = 9.807
# The standard mean sea level air density kg/m** p=1013 hPa, T=20°C
_rho0 = 1.2038631624242195
# Standard dynamic viscosity TODO: shouldn't play huge role, but check it
_mu0 = 1.717696e-5 # Pa*s
# Standard kinematic viscosity
_nu0 = _mu0/_rho0

import numpy as np
from snowScatt._constants import _g as g
from snowScatt._constants import _ice_density as rho_ice
from snowScatt._constants import _rho0
from snowScatt._constants import _nu0



def Boehm1992(diam, mass, area,
			  rho_air=_rho0, nu_air=_nu0, as_ratio=1.0):
	"""
	Bohm, J.(1992).A general hydrodynamic theory for mixed-phase microphysics.
	Part I: Drag and fall speed of hydrometeors.Atmospheric research,27(4), 253–274

	Parameters
	----------
	diam : array(Nparticles) double
		spectrum of diameters of the particles [meters]
	rho_air : scalar double
		air density [kilograms/meter**3]
	nu_air : scalar double
		air kinematic viscosity [meters**2/seconds]
	mass : array(Nparticles) double
		mass of the particles [kilograms]
	area : array(Nparticles) double
		cross section area [meters**2]
	as_ratio : scalar double
		Correction factor for the calculation of area ration in non-spherical
		symmetric particles. See Karrer et al. 2020

	Returns
	-------
	vterm_bohm : array(Nparticles) double
		terminal fallspeed computed according to the model [meters/second]
	"""

	q = area / (np.pi/4.0 * diam**2)
	eta_air = nu_air*rho_air # dynamic viscosity
	
	alpha = np.array(as_ratio) #1.0
	X_0 = 2.8e6
	#X = 8.0*mass*grav*rho_air/(np.pi*(eta_air**2)*q**0.25)
	X = 8.0*mass*g*rho_air/(np.pi*(eta_air**2)*np.maximum(alpha,np.ones_like(alpha)*1.0)*np.maximum(q**0.25,q)) #reduced to 8.0*mtot*grav*rho/(np.pi*(eta**2)*q**(1/4) = 8.0*mtot*grav*rho/(np.pi*(eta**2)*(area / (np.pi/4.0 * diam**2))**(1/4) for alpha=1 and q<1 (which is usually the case)
	k = 1.0 #np.minimum(np.maximum(0.82+0.18*alpha,
			#                  np.ones_like(alpha)*0.85),
			#       0.37+0.63/alpha,
			#       1.33/(np.maximum(np.log(alpha),
			#                        np.ones_like(alpha)*0.0)+1.19)) #k is 1 for alpha=1
	gama_big = np.maximum(np.ones_like(alpha)*1.0, np.minimum(np.ones_like(alpha)*1.98,3.76-8.41*alpha+9.18*alpha**2-3.53*alpha**3)) #1 for alpha=1
	C_DP = np.maximum(0.292*k*gama_big,0.492-0.2/np.sqrt(alpha)) #0.292 for alpha=1
	C_DP = np.maximum(1.0,q*(1.46*q-0.46))*C_DP #0.292 for alpha=1
	C_DP_prim = C_DP*(1.0+1.6*(X/X_0)**2)/(1.0+(X/X_0)**2) #0.292 for small particles; larger for bigger particles 
	beta = np.sqrt(1.0+C_DP_prim/6.0/k*np.sqrt(X/C_DP_prim))-1
	N_Re0 = 6.0*k/C_DP_prim*beta**2
	C_DO = 4.5*k**2*np.maximum(alpha,np.ones_like(alpha)*1.0)
	gama_small = (C_DO - C_DP)/4.0/C_DP
	N_Re  = N_Re0*(1.0 + (2.0*beta*np.exp(-beta*gama_small))/((2.0+beta)*(1.0+beta)) )
	#Re = 8.5*((1.0+0.1519*X**0.5)**0.5-1.0)**2
	vterm_bohm = N_Re*eta_air/diam/rho_air
	return vterm_bohm


def Boehm1989(diam, mass, area,
			  rho_air=_rho0, nu_air=_nu0):
	"""
	Böhm, J.  (1989).A general equation for the terminal fall speed of solid hydrometeors (Vol. 46) (No. 15).
	doi:  10.1175/1520-0469(1989)046〈2419:AGEFTT〉2.0.CO;2915B 
	
	Parameters
	----------
	diam : array(Nparticles) double
		spectrum of diameters of the particles [meters]
	rho_air : scalar double
		air density [kilograms/meter**3]
	nu_air : scalar double
		air kinematic viscosity [meters**2/seconds]
	mass : array(Nparticles) double
		mass of the particles [kilograms]
	area : array(Nparticles) double
		cross section area [meters**2]

	Returns
	-------
	vterm_bohm : array(Nparticles) double
		terminal fallspeed computed according to the model [meters/second]
	"""

	q = area / (np.pi/4.0 * diam**2)
	eta_air = nu_air*rho_air # dynamic viscosity
	
	#alpha = np.array(as_ratio) #1.0
	#X_0 = 2.8e6
	X = 8.0*mass*g*rho_air/(np.pi*(eta_air**2)*q**0.25)
	
	#k = np.minimum(np.maximum(0.82+0.18*alpha,np.ones_like(alpha)*0.85),0.37+0.63/alpha,1.33/(np.maximum(np.log(alpha),np.ones_like(alpha)*0.0)+1.19)) #k is 1 for alpha=1
	#gama_big = np.maximum(np.ones_like(alpha)*1.0, np.minimum(np.ones_like(alpha)*1.98,3.76-8.41*alpha+9.18*alpha**2-3.53*alpha**3)) #1 for alpha=1
	#C_DP = np.maximum(0.292*k*gama_big,0.492-0.2/np.sqrt(alpha)) #0.292 for alpha=1
	#C_DP = np.maximum(1.0,q*(1.46*q-0.46))*C_DP #0.292 for alpha=1
	#C_DP_prim = C_DP*(1.0+1.6*(X/X_0)**2)/(1.0+(X/X_0)**2) #0.292 for small particles; larger for bigger particles 
	#beta = np.sqrt(1.0+C_DP_prim/6.0/k*np.sqrt(X/C_DP_prim))-1
	#N_Re0 = 6.0*k/C_DP_prim*beta**2
	#C_DO = 4.5*k**2*np.maximum(alpha,np.ones_like(alpha)*1.0)
	#gama_small = (C_DO - C_DP)/4.0/C_DP
	#N_Re  = N_Re0*(1.0 + (2.0*beta*np.exp(-beta*gama_small))/((2.0+beta)*(1.0+beta)) )
	Re = 8.5*((1.0+0.1519*X**0.5)**0.5-1.0)**2
	vterm_bohm = Re*eta_air/diam/rho_air
	return vterm_bohm


def HeymsfieldWestbrook2010(diaSpec, mass, area,
							rho_air=_rho0, nu_air=_nu0, k=0.5):
	"""
	Heymsfield, A. J. & Westbrook, C. D. Advances in the Estimation of Ice Particle Fall Speeds
	Using Laboratory and Field Measurements. Journal of the Atmospheric Sciences 67, 2469-2482 (2010).
	equations 9-11

	Parameters
	----------
	diaSpec : array(Nparticles) double
		spectrum of diameters of the particles [meters]
	rho_air : scalar double
		air density [kilograms/meter**3]
	nu_air : scalar double
		air kinematic viscosity [meters**2/seconds]
	mass : array(Nparticles) double
		mass of the particles [kilograms]
	area : array(Nparticles) double
		cross section area [meters**2]
	k : scalar double
		tuning coefficient for turbulent flow defaults to 0.5

	Returns
	-------
	velSpec : array(Nparticles) double
		terminal fallspeed computed according to the model [meters/second]
	"""

	delta_0 = 8.0
	C_0 = 0.35
	
	area_proj = area/((np.pi/4.)*diaSpec**2) # area ratio
	eta = nu_air * rho_air #now dynamic viscosity

	Xstar = 8.0*rho_air*mass*g/(np.pi*area_proj**(1.0-k)*eta**2)# !eq 9
	Re=0.25*delta_0**2*((1.0+((4.0*Xstar**0.5)/(delta_0**2.0*C_0**0.5)))**0.5 - 1 )**2 #!eq10
	 
	velSpec = eta*Re/(rho_air*diaSpec)
	return velSpec


def KhvorostyanovCurry2005(diam, mass, area,
						   rho_air=_rho0, nu_air=_nu0, smooth=False):
	"""
	Khvorostyanov, V. I., & Curry, J. A.    (2005).    Fall velocities of hydrometeors in the atmosphere:
	Refinements to a continuous analytical power law.Journal of the Atmospheric Sciences,62(12), 4343-4357.
	doi:  10.1175/JAS3622.1

	Parameters
	----------
	diam : array(Nparticles) double
		spectrum of diameters of the particles [meters]
	rho_air : scalar double
		air density [kilograms/meter**3]
	nu_air : scalar double
		air kinematic viscosity [meters**2/seconds]
	mass : array(Nparticles) double
		mass of the particles [kilograms]
	area : array(Nparticles) double
		cross section area [meters**2]
	smooth : scalar bool
		Decide wheather or not use the smooth approximation for the estimation
		of the drag coefficient from the Best number X

	Returns
	-------
	velSpec : array(Nparticles) double
		terminal fallspeed computed according to the model [meters/second]
	"""

	# Best number eq. (2.4b) with buoyancy
	Vb = mass/rho_ice
	Fb = rho_air * Vb * g
	eta_air = nu_air*rho_air # dynamic viscosity
	Xbest = 2. * np.abs(mass*g-Fb) * rho_air * diam**2 / (area * eta_air**2)
	if( smooth ):
	  Cd  = X2Cd_kc05smooth(Xbest)
	else:
	  Cd  = X2Cd_kc05rough(Xbest)
	return np.sqrt( 2*np.abs(mass*g - Fb)/(rho_air * area * Cd))


def X2Cd_kc05rough(Xbest):
	do_i = 5.83
	co_i = 0.6
	Ct = 1.6
	X0_i = 0.35714285714285714285e-6 # 1.0/2.8e6
	# derived constants
	c1 = 4.0 / ( do_i**2 * np.sqrt(co_i))
	c2 = 0.25 * do_i**2
	# Re-X eq. (2.5)
	bracket = np.sqrt(1.0 + c1 * np.sqrt(Xbest)) - 1.0
	# turbulent Reynold's number, eq (3.3)
	psi = (1.0+(Xbest*X0_i)**2) / (1.0+Ct*(Xbest*X0_i)**2)
	Re  = c2*bracket**2 # * np.sqrt(psi) # TODO remove psi in Re?
	# eq. (2.1) from KC05 with (3.2)
	return co_i * (1.0 + do_i / np.sqrt(Re))**2 / psi


def X2Cd_kc05smooth(Xbest):
	do_i = 9.06
	co_i = 0.292
	Ct = 1.6
	X0_i = 1.0/6.7e6
	c1 = 4.0/(do_i**2 * np.sqrt(co_i))
	c2 = 0.25 * do_i**2
	# Re-X eq. (2.5)
	bracket = np.sqrt(1.0 + c1 * np.sqrt(Xbest)) - 1.0
	# turbulent Reynold's number, eq (3.3)
	psi = (1+(Xbest*X0_i)**2) / (1+Ct*(Xbest*X0_i)**2)
	Re  = c2*bracket**2 #* np.sqrt(psi) # TODO remove psi in Re?
	# eq. (2.1) from KC05 with (3.2)
	return co_i * (1. + do_i/np.sqrt(Re))**2 / psi

if __name__ == "__main__":

    diam = 1e-3 #diameter 
    rho =  200     #density [kg/m^3]
    mass = np.pi/5*rho*diam**3
    area = np.pi/4*diam**2
    
    vel = Boehm1992(diam, mass, area,rho_air=_rho0, nu_air=_nu0, as_ratio=1.0)

    print(vel)
