import numpy as np

def transformPSD(a=np.nan,b=np.nan,nu=np.nan,mu=np.nan,xmin=np.nan,xmax=np.nan,A=np.nan,lam=np.nan):
    """
    This is a converter from the Axel Seifert MAP (2006) ice parametrization scheme to pamtra
    Anything that is not set or not computable goes to nan
    """
    bm  = 1.0/b
    am  = 1.0/a**bm
    N0  = am**(nu+1.0)*bm*A
    mum = nu*bm+bm-1.0
    LAM = lam*am**mu
    gam = bm*mu
    Dmin= a*xmin**b
    Dmax= a*xmax**b

    print('am = ',  am)
    print('bm = ',  bm)
    print('N0 = ',  N0)
    print('mum= ', mum)
    print('LAM= ', LAM)
    print('gam= ', gam)
    print('Dmn= ',Dmin)
    print('Dmx= ',Dmax)

def getVelocitySize(ag=np.nan,bg=np.nan,av=np.nan,bv=np.nan):
    """
    This is a converter from the Axel Seifert MAP (2006) ice parametrization scheme to pamtra
    Anything that is not set or not computable goes to nan
    """
    beta = bv/bg
    alpha = av/ag**beta
    print('alpha_v = ',alpha)
    print('beta_v  = ',beta)
    bm = 1.0/bg
    am = 1.0/ag**bm
    print('am = ',  am)
    print('bm = ',  bm)


print('snow_cosmo5')
transformPSD(a=2.4,b=0.455,nu=0.0,mu=0.5,xmin=1.0e-10,xmax=2.0e-5)
getVelocitySize(ag=2.4,bg=0.455,av=8.8,bv=0.15)

print('\n snow_SBB')
transformPSD(a=5.13,b=0.5,nu=0.0,mu=0.5,xmin=1.0e-10,xmax=2.0e-5)
getVelocitySize(ag=5.13,bg=0.5,av=8.294,bv=0.125)

print('\n snow_SBB_nonsphere')
transformPSD(a=5.13,b=0.5,nu=0.0,mu=1.0/3.0,xmin=1.0e-10,xmax=2.0e-5)
getVelocitySize(ag=5.13,bg=0.5,av=np.nan,bv=np.nan)

print('\n graupelhail_cosmo5')
transformPSD(a=1.42e-1,b=0.314,nu=1.0,mu=1.0/3.0,xmin=1.0e-9,xmax=5.0e-4)
getVelocitySize(ag=1.42e-1,bg=0.314,av=86.89371,bv=0.268325)

print('\n hail_cosmo5')
transformPSD(a=0.1366,b=1.0/3.0,nu=1.0,mu=1.0/3.0,xmin=2.6e-9,xmax=5.0e-4)
getVelocitySize(ag=0.1366,bg=1.0/3.0,av=39.3,bv=1.0/6.0)

print('\n ice_cosmo5')
transformPSD(a=0.835,b=0.39,nu=0.0,mu=1.0/3.0,xmin=1.0e-12,xmax=1.0e-5)
getVelocitySize(ag=0.835,bg=0.39,av=27.7,bv=0.21579)

print('\n cloud_cosmo5')
transformPSD(a=1.24e-1,b=1.0/3.0,nu=1.0,mu=1.0,xmin=4.2e-15,xmax=2.6e-10)
getVelocitySize(ag=1.24e-1,bg=1.0/3.0,av=3.75e5,bv=2.0/3.0)

print('\n ice_cosmo5_nonsphere')
transformPSD(a=0.835,b=0.39,nu=0.0,mu=1.0/3.0,xmin=1.0e-14,xmax=1.0e-5)
getVelocitySize(ag=0.835,bg=0.39,av=np.nan,bv=np.nan)

print('\n ice_cosmo5xmin-14')
transformPSD(a=0.835,b=0.39,nu=0.0,mu=1.0/3.0,xmin=1.0e-14,xmax=1.0e-5)
getVelocitySize(ag=0.835,bg=0.39,av=27.7,bv=0.21579)

print('\n ice_hex')
transformPSD(a=0.22,b=1.0/3.31,nu=0.0,mu=1.0/3.0,xmin=1.0e-14,xmax=1.0e-5)
getVelocitySize(ag=0.22,bg=1.0/3.31,av=41.9,bv=0.26)

print('\n cloud_cosmo')
transformPSD(a=0.124,b=1.0/3.0,nu=0.0,mu=1.0/3.0,xmin=4.2e-15,xmax=2.6e-10)
getVelocitySize(ag=0.124,bg=1.0/3.0,av=3.75e5,bv=2.0/3.0)

print('\n rain_SBB')
transformPSD(a=0.124,b=1.0/3.0,nu=0.0,mu=1.0/3.0,xmin=2.6e-10,xmax=3.0e-6)
getVelocitySize(ag=0.124,bg=1.0/3.0,av=114.0137,bv=0.23437)

print('\n dendrite')
transformPSD(a=5.17,b=1.0/2.29,nu=0.0,mu=1.0/3.0,xmin=1.0e-13,xmax=1.0e-5)
getVelocitySize(ag=5.17,bg=1.0/2.29,av=11.0,bv=0.21)


#rho = lambda D: (6.0*1.588/np.pi)*D**(2.56-3)
#rho440 = lambda x: 440.0 + 0.0*x
#F94 = lambda x: 84.0 + 0.0*x
#rho144 = lambda x: 140.0 + 0.0*x
#Hey2002 = lambda D: 1000*0.0265*(D*100.0)**(-0.46)
#F94 = lambda x: 1000*0.044*6/np.pi + 0.0*x
#D = np.linspace(1e-5,1e-3,1000)
#import matplotlib.pyplot as plt
#plt.close('all')
#plt.plot(D*1000.0,rho(D),label='Seifert 2m model ice')
#plt.plot(D*1000.0,rho440(D),label='Constant density 440 kg/m3')
#plt.plot(D*1000.0,Hey2002(D),label='Heymsfield (2002) 5b-rosette samples')
#plt.plot(D*1000.0,F94(D),label='Ferrier (1994) 4ICE')
#plt.plot(D*1000.0,rho144(D),label='Ferrier (1994) C3')
#plt.grid()
#plt.legend()
#plt.xlabel('particle size [mm]')
#plt.ylabel('density [kg/m3]')

size = np.arange(0.0001,0.004,0.0001)
import matplotlib.pyplot as plt
plt.figure(figsize=(7,3), dpi=300)
#plt.figure()
#plt.plot(size,154.214*np.power(size,0.8606), label='ice_hex')
plt.plot(1e3*size,30.6063*np.power(size,0.5533), label='ice') #cosmo
plt.plot(1e3*size,5.51105*np.power(size,0.2500), label='snow') #SBB
plt.plot(1e3*size,460.67*np.power(size,0.8545), label='graupel')
#plt.plot(size,powLaw(size,243388657.6,2.0), label='cloud')
plt.xlabel('size   [mm]')
plt.ylabel('fallspeed    [m/s]')
plt.grid()
plt.legend()

plt.figure()
plt.semilogy(size,1.5878*np.power(size,2.564), label='ice') # cosmo
plt.semilogy(size,0.038*np.power(size,2.00), label='snow') #SBB
plt.semilogy(size,500.86*np.power(size,3.1847), label='graupel')
#plt.plot(size,powLaw(size,243388657.6,2.0), label='cloud')
plt.xlabel('size   [m')
plt.ylabel('mass    [kg]')
plt.grid()
plt.legend()

plt.figure()
plt.loglog(size,1.5878*np.power(size,2.564), label='ice') # cosmo
plt.loglog(size,0.038*np.power(size,2.00), label='snow') #SBB
plt.loglog(size,500.86*np.power(size,3.1847), label='graupel')
#plt.plot(size,powLaw(size,243388657.6,2.0), label='cloud')
plt.xlabel('size   [m]')
plt.ylabel('mass    [kg]')
plt.grid()
plt.legend()

