'''
query particle properties in the SB-scheme and from Karrer et al. (2020)
just run the script and look around in the dictionary called "p"
e.g. $ python pprop_dict.py 
ipdb> p["Mix2"].a_Atlas #get a-coefficient in Atlas type

additionally you can also calculate the PSD/or PMD (mass)
'''

import numpy as np
#import pyPamtra
import sys

from IPython.core.debugger import set_trace
run_quick=False

#print("Settings")
#replaced cloud ice by "Column" and snow by "Mix2" #ATTENTION: size distribution parameter are not recalculated according 
#define particle class
def init_class():
    class particle(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
#define objects for particle types
    p = dict()
    p["column"] = particle(     
            nu_SB	=  0.0, #shape parameter is mass distribution N(m)= A * m**nu_SB * exp(- lam m ** mu_SB)
            mu_SB       = 0.333, #shape parameter is mass distribution N(m)= A * m**nu_SB * exp(- lam m ** mu_SB)
            a_geo       =  (1./0.046)**(1./2.07), #coefficient in size-mass relation Dmax = a_geo * m ** b_geo
            b_geo       =  1./2.07,#coefficient in size-mass relation Dmax = a_geo * m ** b_geo
            xmax        =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
            xmin        =  1.00e-12, #& !..x_min..minimale Teilchenmasse D=200e-6m
            a_Atlas     =1.629, #coefficient in Atlas-type velocity relation vterm = a_Atlas - b_Atlas * exp( -c_Atlas * D(mass equivalent diameter) )
            b_Atlas     =1.667,#coefficient in Atlas-type velocity relation vterm = a_Atlas - b_Atlas * exp( -c_Atlas * D(mass equivalent diameter) )
            c_Atlas     =1586.0,#coefficient in Atlas-type velocity relation vterm = a_Atlas - b_Atlas * exp( -c_Atlas * D(mass equivalent diameter) )
            gam_eq      =8.212834,  # coefficient in projected area relation A(D_eq) = gam_eq * D(mass equivalent diameter)** sig_eq
            sig_eq      =2.23     )  # coefficient in projected area relation A(D_eq) = gam_eq * D(mass equivalent diameter)** sig_eq

    p["column_narrow"] = particle(      nu_SB	=  2.0,
            mu_SB       = 0.333,
            a_geo       =  (1./0.046)**(1./2.07),
            b_geo       =  1./2.07,
            xmax        =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
            xmin        =  1.00e-12, #& !..x_min..minimale Teilchenmasse D=200e-6m
            a_Atlas     =1.629,
            b_Atlas     =1.667,
            c_Atlas     =1586.0,
            gam_eq      =8.212834,  #& !..gam_eq, A(D_eq) (not set here!) 
            sig_eq      =2.23     ) #!..sig_eq, A(D_eq) (not set here!)
    p["plate"] = particle(      nu_SB	=  0.,
            mu_SB       = 0.3333,
            a_geo       =  (1./0.788)**(1./2.48),
            b_geo       =  1./2.48,
            xmax        =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
            xmin        =  1.00e-12, #& !..x_min..minimale Teilchenmasse D=200e-6m
            a_Atlas     =2.265,
            b_Atlas     =2.275,
            c_Atlas     =771.138)
    p["dendrite"] = particle(    nu_SB	=  0.,
            mu_SB       = 0.3333,
            a_geo       =  (1./0.074)**(1./2.33),
            b_geo       =  1./2.33,
            xmax        =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
            xmin        =  1.00e-12, #& !..x_min..minimale Teilchenmasse D=200e-6m
            a_Atlas     =1.133,
            b_Atlas     =1.153,
            c_Atlas     =1177.0)
    p["needle"] = particle(      nu_SB       =  0.,
            mu_SB       = 0.3333,
            a_geo       =  (1./0.005)**(1./1.89),
            b_geo       =  1./1.89,
            xmax        =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
            xmin        =  1.00e-12, #& !..x_min..minimale Teilchenmasse D=200e-6m
            a_Atlas     = 0.848,
            b_Atlas     = 0.871,
            c_Atlas     = 2278.0,
            gam_eq      = 13.96877,  #& !..gam_eq, A(D_eq) (not set here!) 
            sig_eq      = 2.258579     ) #!..sig_eq, A(D_eq) (not set here!)
    '''
    = particle(       nu_SB	=  0.,
            mu_SB       = 0.3333,
            a_geo       =  (1./)**(1./),
            b_geo       =  1./,
            xmax        =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
            xmin        =  1.00e-12, #& !..x_min..minimale Teilchenmasse D=200e-6m
            a_Atlas     =,
            b_Atlas     =,
            c_Atlas     =)
    '''
#,
    p["aggPlate"] = particle(    nu_SB     =  0.,
            mu_SB       = 0.5,
            a_geo       =  (1./0.076)**(1./2.22),
            b_geo       =  1./2.22,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_Atlas     =1.366,
            b_Atlas     =1.391,
            c_Atlas     =1285.6,
            ssrg        ="0.18_0.89_2.06_0.08")
    p["aggDendrite"] = particle( nu_SB     =  0.,
            mu_SB       = 0.5,
            a_geo       =  (1./0.027)**(1./2.22),
            b_geo       =  1./2.22,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_Atlas     =0.880,
            b_Atlas     =0.895,
            c_Atlas     =1393.0,
            ssrg        ="0.23_0.75_1.88_0.10")
    p["aggNeedle"] = particle(         nu_SB     =  0.,
            mu_SB       = 0.5,
            a_geo       =  (1./0.028)**(1./2.11),
            b_geo       =  1./2.11,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_Atlas     =1.118,
            b_Atlas     =1.133,
            c_Atlas     =1659.5,
            ssrg        ="0.25_0.76_1.66_0.04")
    p["aggColumn"] = particle(   nu_SB     =  0.,
            mu_SB       = 0.5,
            a_geo       =  (1./0.074)**(1./2.15),
            b_geo       =  1./2.15,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_Atlas     =1.583,
            b_Atlas     =1.600,
            c_Atlas     =1419.2,
            ssrg        ="0.23_1.45_2.05_0.02")
    p["Mix1"] = particle(     nu_SB     =  0.,
            mu_SB       = 0.5,
            a_geo       =  (1./0.045)**(1./2.16),
            b_geo       =  1./2.16,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_Atlas     =1.233,
            b_Atlas     =1.250,
            c_Atlas     =1509.5)
    p["Mix2"] = particle(      nu_SB	=  0.0,
            mu_SB       = 0.3333,
            a_geo       =  (1./0.017)**(1./1.95),
            b_geo       =  1./1.95,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_Atlas     =1.121,
            b_Atlas     =1.119,
            c_Atlas     =2292.2,
            a_vel       = 1.964e01,
            b_vel       = 0.202645,
            ssrg        ="0.22_0.60_1.81_0.11",
            gam_eq      =685.93,   # & !..gam_eq, A(D_eq)
            sig_eq      =2.73      ) #!..sig_eq, A(D_eq)
    p["Mix2_narrow"] = particle(      nu_SB	=  2.0,
            mu_SB       = 0.3333,
            a_geo       =  (1./0.017)**(1./1.95),
            b_geo       =  1./1.95,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_Atlas     =1.121,
            b_Atlas     =1.119,
            c_Atlas     =2292.2,
            ssrg        ="0.22_0.60_1.81_0.11")
    p["Mix2SB"] = particle(      nu_SB	=  0.,
            mu_SB       = 0.5,
            a_geo       =  5.13, #(1./0.017)**(1./1.95),
            b_geo       =  1./2,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_Atlas     =1.121,
            b_Atlas     =1.119,
            c_Atlas     =2292.2,
            ssrg        ="0.23_0.60_1.8_0.11",
            gam_eq      =685.93,   # & !..gam_eq, A(D_eq)
            sig_eq      =2.73      ) #!..sig_eq, A(D_eq)

    p["cloud_nuemue1"] = particle(nu_SB	=  1.,
                            mu_SB       = 1.0,
                            a_geo       =  0.124,
                            b_geo       =  0.33333,
                            xmax       = 2.60e-10, #& !..x_max..maximale Teilchenmasse D=80e-6m
                            xmin       = 4.20e-15) #& !..x_min..minimale Teilchenmasse D=2.e-6m
    p["SBB_rain"]	 = particle(nu_SB		=  0.0,
                            mu_SB        = 0.33333,
                            a_geo      =  0.124,
                            b_geo      =  0.33333,
                            a_Atlas     =  9.292000,  #& !..alfa
                            b_Atlas     =  9.623000,  #& !..beta
                            c_Atlas     =  6.222e+2,  #& !..gama
                            xmax      = 3.00e-06,  #& !..x_max
                            xmin      = 2.60e-10)  #& !..x_min
    p["SBB_cloud_ice"] = particle(nu_SB	        =  0.,
                            mu_SB        = 0.3333,
                            a_geo      =  0.835,
                            b_geo      =  0.39,
                            a_vel  = 2.60e+01, #coefficient in powerlaw relation of the terminal velocity vterm=a_vel*m**b_vel
                            b_vel  = 0.215790, #coefficient in powerlaw relation of the terminal velocity vterm=a_vel*m**b_vel
                            xmax       =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
                            xmin       =  1.00e-12) #& !..x_min..minimale Teilchenmasse D=200e-6m
    p["SBB_snow"]      = particle(   nu_SB=  0.0,
                            mu_SB = 0.5,
                            a_geo  =  5.13,
                            b_geo  =  0.5,
                            a_vel  =  8.294 ,
                            b_vel  =  0.125 ,
                            xmax   =  2.00e-05, #& !..x_max..maximale Teilchenmasse
                            xmin   =  1.00e-10) #& !..x_min..minimale Teilchenmasse
    p["SBB_graupel"] = particle(     nu_SB      =  1.0, #graupelhail_cosmo5
                            mu_SB      =  0.33333,
                            a_geo      =  0.142,
                            b_geo      =  0.314,
                            a_vel      =  86.89371,
                            b_vel      =  0.268325,
                            xmax       =  5.00e-04, #& #!..x_max..maximale Teilchenmasse
                            xmin       =  1.00e-09) #& !..x_min..minimale Teilchenmasse
    p["SBB_graupel_denseX2"] = particle(     nu_SB      =  1.0, #graupelhail_cosmo5 _denseX2 = particle_frozen( & ! graupelhail2test4 but two times lower density
                            mu_SB      =  0.33333,
                            a_geo      =  0.176,
                            b_geo      =  0.314,
                            a_vel      =  86.89371,
                            b_vel      =  0.268325,
                            xmax       =  5.00e-04, #& #!..x_max..maximale Teilchenmasse
                            xmin       =  1.00e-09) #& !..x_min..minimale Teilchenmasse
    p["SBB_graupel_denseX4"] = particle(     nu_SB      =  1.0, #graupelhail_cosmo5 _denseX2 = particle_frozen( & ! graupelhail2test4 but two times lower density
                            mu_SB      =  0.33333,
                            a_geo      =  0.220,
                            b_geo      =  0.314,
                            a_vel      =  86.89371,
                            b_vel      =  0.268325,
                            xmax       =  5.00e-04, #& #!..x_max..maximale Teilchenmasse
                            xmin       =  1.00e-09) #& !..x_min..minimale Teilchenmasse
    p["SBB_hail"] = particle(        nu_SB      =  1.0,
                            mu_SB      =  0.33333,
                            a_geo      =  0.1366,
                            b_geo      =  0.3333333,
                            xmax       =  5.00e-04, #& !..x_max..maximale Teilchenmasse
                            xmin       =  2.60e-9) #& !..x_min..minimale Teilchenmasse
    '''
    p["agg"] = particle(         nu_SB     =  0.,
            mu_SB       = 0.333,
            a_geo       =  (1./)**(1./),
            b_geo       =  1./,
            xmax        =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
            xmin        =  1.00e-12, #& !..x_min..minimale Teilchenmasse D=200e-6m
            a_Atlas     =,
            b_Atlas     =,
            c_Atlas     =)
    '''
#convert from N(m) to N(D) space
    for key in p.keys():
        curr_cat=p[key]
#for curr_cat in [cloud_ice,snow]:
        curr_cat.a_ms = (1./curr_cat.a_geo)**(1./curr_cat.b_geo)
        curr_cat.b_ms = 1./curr_cat.b_geo
        curr_cat.mu =  curr_cat.b_ms*curr_cat.nu_SB+curr_cat.b_ms-1
        curr_cat.gam = curr_cat.b_ms*curr_cat.mu_SB
    return p

def calculate_PSD(twomom,curr_cat,diam,i_height=249):
    '''
    calculate the normalized number concentration (as a function of diameter) corresponding to moments of a self-defined category 
    INPUT:  twomom: dictionary containing the moments
            curr_cat: class containing particle properties
            diam: diameter array at which the number concentration should be evaluated
            i_height: height index of the entries in the twomom dictionary which should be analyzed
    '''

    from scipy.special import gamma
    ###
    #calculate the normalized number concentration 
    ###
    #calculate bin width (del_diam)
    del_diam = np.diff(diam)
    diam_2 = diam[:-1] #diameter without last element

    #copy the mass density and the number concentration to PAMTRA conventions
    q_h  =  twomom[curr_cat.mixrat_var][i_height]
    n_tot =  twomom[curr_cat.numcon_var][i_height]

    #calculate the distribution based on PAMTRA code (taken from PAMTRA make_dist_params.f90)
    work2 = gamma((curr_cat.mu + curr_cat.b_ms + 1.0) / curr_cat.gam)
    work3 = gamma((curr_cat.mu + 1.0) / curr_cat.gam)
    lam	=	(curr_cat.a_ms / q_h * n_tot * work2 / work3)**(curr_cat.gam / curr_cat.b_ms)
    N_0 = curr_cat.gam * n_tot / work3 * lam**((curr_cat.mu + 1.0) / curr_cat.gam)

    N_D	= N_0*diam_2**curr_cat.mu*np.exp(-lam*diam_2**curr_cat.gam) #not normalized number concentrations; normalized with /del_diam
    #N_D	= N_0*diam_2**curr_cat.mu*np.exp(-lam*diam_2**curr_cat.gam)*del_diam #normalized number concentration
    M_D = N_D * curr_cat.a_ms*diam_2**curr_cat.b_ms
    
    return N_D,M_D

def calculate_PSD2(q,N,curr_cat,Dmax_array):
    '''
    q,N:        0th and 1st moment of the particle mass distribution (PSM)
    curr_cat:   particle class (contains shape parameter etc.)
    Dmax_array: array of Dmax at which PSD is evaluated
    '''
    from scipy.special import gamma

    #calculate the distribution based on PAMTRA code (taken from PAMTRA make_dist_params.f90)
    work2 = gamma((curr_cat.mu_max + curr_cat.b_ms + 1.0) / curr_cat.gam)
    work3 = gamma((curr_cat.mu_max + 1.0) / curr_cat.gam)
    lam	=	(curr_cat.a_ms / q * N * work2 / work3)**(curr_cat.gam / curr_cat.b_ms)
    N_0 = curr_cat.gam * N / work3 * lam**((curr_cat.mu_max + 1.0) / curr_cat.gam)
    
    N_D	= N_0*Dmax_array**curr_cat.mu_max*np.exp(-lam*Dmax_array**curr_cat.gam) #/Dmax_array #not normalized number concentrations; normalized with /del_diam

    return N_D

def calculate_PSM(q,N,curr_cat,mass_array):
    '''
    q,N:        0th and 1st moment of the particle mass distribution (PSM)
    curr_cat:   particle class (contains shape parameter etc.)
    mass_array: array of masses at which PSM is evaluated
    '''
    from scipy.special import gamma

    lam = (gamma((curr_cat.nu_SB+1.)/curr_cat.mu_SB)/gamma((curr_cat.nu_SB+2.)/curr_cat.mu_SB)*q/N)**(-curr_cat.mu_SB)
    A   = curr_cat.mu_SB * N / gamma((curr_cat.nu_SB+1.)/curr_cat.mu_SB) * lam**((curr_cat.nu_SB+1.)/curr_cat.mu_SB)
    
    f_m = A * mass_array**curr_cat.nu_SB * np.exp(-lam * mass_array**curr_cat.mu_SB) 

    return f_m

def calculate_moments_and_fluxes_from_PSM(mass_array,f_m,curr_cat):
    '''
    curr_cat:   particle class (contains shape parameter etc.)
    mass_array: array of masses at which PSM is evaluated
    f_m:        mass distribution
    '''
    #TODO?: fluxes are not calculated yet

    moments = [0,0,0]
    fluxes  = [0,0,0]

    delta_m_array = np.diff(mass_array)

    for i_mom in [0,1,2]:
        for i_mass,mass in enumerate(mass_array[:-1]):
            moments[i_mom] += (mass_array[i_mass+1]**i_mom*f_m[i_mass+1] + mass_array[i_mass]**i_mom*f_m[i_mass])/2. * delta_m_array[i_mass]
            #fluxes[i_mom] += (mass_array[i_mass+1]**i_mom*f_m[idd_mass+1]*curr_cat.a_Atlas-curr_cat.b_Atlas*np.exp(-curr_cat.c_Atlas* + mass_array[i_mass]**i_mom*f_m[i_mass])/2. * delta_m_array[i_mass]


    return moments

def conv_from_Nm_to_ND(curr_cat,nu_SB=None,mu_SB=None):
   '''
   convert from N(m) to N(D)
   ARGUMENTS:
   curr_cat: a "particle"-object (e.g. p["Mix2"])
   nu_SB: modification of the nu_SB parameter
   mu_SB: modification of the mu_SB parameter
   '''
   if nu_SB!=None:
       curr_cat.nu_SB = nu_SB
   if mu_SB!=None:
       curr_cat.mu_SB = mu_SB
   #convert from N(m) to N(D) space
   curr_cat.a_ms = (1./curr_cat.a_geo)**(1./curr_cat.b_geo)
   curr_cat.b_ms = 1./curr_cat.b_geo
   curr_cat.mu =  curr_cat.b_ms*curr_cat.nu_SB+curr_cat.b_ms-1
   curr_cat.gam = curr_cat.b_ms*curr_cat.mu_SB
   return curr_cat

def calculate_vD_Atlas(cat,Dmax):

    rho     = 1000. #water bulk density
    Deq     = (6.*cat.a_ms/np.pi/rho*Dmax**cat.b_ms)**(1./3.)

    vterm   = cat.a_Atlas-cat.b_Atlas*np.exp(-cat.c_Atlas*Deq)

    return vterm

def main(particle_types,nu_SB_array=None,mu_SB_array=None):
    p= init_class()
    for particle in particle_types:
        print("calculate: ",particle    )
        #select particle type
        if particle=="plate":
            cloud_ice=p["plate"]
            snow=p["aggPlate"]
        elif particle=="needle":
            cloud_ice=p["needle"]
            snow=p["aggNeedle"]
        elif particle=="dendrite":
            cloud_ice=p["dendrite"]
            snow=p["aggDendrite"]
        elif particle=="column":
            cloud_ice=p["column"]
            snow=p["aggColumn"]
        elif particle=="col_Mix1":
            cloud_ice=p["column"]
            snow=p["Mix1"]
        elif particle=="col_Mix2":
            cloud_ice=p["column"]
            snow=p["Mix2"]
        elif particle=="col_Mix2SB":
            cloud_ice=p["column"]
            snow=p["Mix2"]
        elif particle=="cloud_nuemue1":
            cloud_ice   = p["cloud_nuemue1"] #thats a hack -> we only want to get cloud water here
            snow        = cloud_ice

        if nu_SB_array==None:
            nu_SB_array=[snow.nu_SB]
        if mu_SB_array==None:
            mu_SB_array=[snow.mu_SB]
        for nu_SB in nu_SB_array: 
            for mu_SB in mu_SB_array:
                print(nu_SB,mu_SB)
                #recalculate mu based on the set nu_SB
                cloud_ice.mu    =   cloud_ice.b_ms*nu_SB+cloud_ice.b_ms-1
                snow.mu         =   snow.b_ms*nu_SB+snow.b_ms-1
                #recalculate gam based on the set mu_SB
                cloud_ice.gam   =   cloud_ice.b_ms*mu_SB
                snow.gam        =   snow.b_ms*mu_SB
                if not particle in ["cloud_nuemue1"]:
                    print(nu_SB,mu_SB,"particle category a_ms,  b_ms,  mu,  gam, Atlas a , b , c")
                    print(particle,"cloud ice",cloud_ice.a_ms,cloud_ice.b_ms,cloud_ice.mu,cloud_ice.gam,cloud_ice.a_Atlas,cloud_ice.b_Atlas,cloud_ice.c_Atlas)
                    print(particle,"snow",snow.a_ms,snow.b_ms,snow.mu,snow.gam,snow.a_Atlas,snow.b_Atlas,snow.c_Atlas)
                else:
                    print(nu_SB,mu_SB,"particle category a_ms,  b_ms,  mu,  gam")
                    print(particle,particle,cloud_ice.a_ms,cloud_ice.b_ms,cloud_ice.mu,cloud_ice.gam)

if __name__ == "__main__":
    p= init_class()

    ###get some graupel properties
    #g=p["SBB_graupel_denseX2"]
    #g0=p["SBB_graupel"]

    ###illustrate how to calculate the terminal velocity
    for cat in ["plate"]:
        curr_cat = p[cat]
        Dmax    = np.logspace(-4,np.log10(4e-2),100)
        vterm   =   calculate_vD_Atlas(curr_cat,Dmax)
        print(cat,"vterm",vterm)
    import matplotlib.pyplot as plt
    plt.semilogx(Dmax,vterm)
    plt.show()
    set_trace()

    ###calculate some parameters needed for PAMTRA calculations (e.g. PSD parameters in N(D))
    #main(["column","needle","col_Mix2"],nu_SB_array=[0.0,2.0],mu_SB_array=None)
    #main(["col_Mix2"],nu_SB_array=[-0.4845361,0.0,2.0],mu_SB_array=None)
    #main(["SBB_graupel_denseX2"])

