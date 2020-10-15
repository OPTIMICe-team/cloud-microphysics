'''
check implementation of the bulk ventilation coefficient routines in the SB scheme
see also 4.4.2 in Axel's diss "Parametrisierung wolkenmikrophysikalische und Simulation konvektiver Mischwolken
and Appending in Seifert 2008, "On the evaporation ..."
'''

import numpy as np
from scipy.special import gamma
from IPython.core.debugger import Tracer ; debug = Tracer()
import matplotlib.pyplot as plt

#define particle class
def init_class():
    class particle(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    #define objects for particle types
    p = dict()

    p["column"] = particle(     
            nu	        =  0.0, #shape parameter is mass distribution N(m)= A * m**nu_SB * exp(- lam m ** mu_SB)
            mu          =  0.333, #shape parameter is mass distribution N(m)= A * m**nu_SB * exp(- lam m ** mu_SB)
            a_geo       =  (1./0.046)**(1./2.07), #coefficient in size-mass relation Dmax = a_geo * m ** b_geo
            b_geo       =  1./2.07,#coefficient in size-mass relation Dmax = a_geo * m ** b_geo
            xmax        =  1.00e-05, #& !..x_max..maximale Teilchenmasse D=???e-2m
            xmin        =  1.00e-12, #& !..x_min..minimale Teilchenmasse D=200e-6m
            a_vel       =  49.5085179179, 
            b_vel       =  0.251465669485,
            a_Atlas     =  1.667, #ATTENTION: this is not the fitted coefficients, but a_Atlas<b_Atlas #a_Atlas = 1.629, #coefficient in Atlas-type velocity relation vterm = a_Atlas - b_Atlas * exp( -c_Atlas * D(mass equivalent diameter) )
            b_Atlas     =  1.667,#coefficient in Atlas-type velocity relation vterm = a_Atlas - b_Atlas * exp( -c_Atlas * D(mass equivalent diameter) )
            c_Atlas     =  1586.0,#coefficient in Atlas-type velocity relation vterm = a_Atlas - b_Atlas * exp( -c_Atlas * D(mass equivalent diameter) )
            a_ven       =  0.86,
            b_ven       =  0.28,
            gam_eq      = 8.212834,  # coefficient in projected area relation A(D_eq) = gam_eq * D(mass equivalent diameter)** sig_eq
            sig_eq      = 2.23     )  # coefficient in projected area relation A(D_eq) = gam_eq * D(mass equivalent diameter)** sig_eq

    p["Mix2"] = particle(      
            nu  	=  0.0,
            mu          = 0.3333,
            a_geo       =  (1./0.017)**(1./2.0), #(1./0.017)**(1./1.95),
            b_geo       =  1./2.0, #1./1.95,
            xmax        =  2.00e-05, #& !..x_max..minimale Teilchenmasse 
            xmin        =  1.00e-10, #& !..x_min..minimale Teilchenmasse
            a_vel       =  1.964e1,
            b_vel       =  0.202645,
            a_Atlas     =  1.121,
            b_Atlas     =  1.119,
            c_Atlas     =  2292.2,
            a_ven       =  0.86,
            b_ven       =  0.28,
            ssrg        =  "0.22_0.60_1.81_0.11",
            gam_eq      =  685.93,   # & !..gam_eq, A(D_eq)
            sig_eq      =  2.73      ) #!..sig_eq, A(D_eq)

    p["rain_mD"] = particle(      
            nu  	=  0.0,
            mu          = 0.3333,
            a_geo       =  1.24e-01,
            b_geo       =  0.333333,
            a_vel       =  1.964e1,
            b_vel       =  0.202645,
            a_Atlas     =  1.121,
            b_Atlas     =  1.119,
            c_Atlas     =  2292.2,
            a_ven       =  0.86,
            b_ven       =  0.28,
            ssrg        =  "0.22_0.60_1.81_0.11",
            gam_eq      =  685.93,   # & !..gam_eq, A(D_eq)
            sig_eq      =  2.73      ) #!..sig_eq, A(D_eq)
    return p


def generalized_gamma(n,x,nu,mu,x_array): 
    '''
    calculate the generalized gamma distribution with mass coordinate
    INPUT:
        n: number density
        x: mean mass
        nu,mu: shape parameters
        x_array: grid to evaluate PSD
    OUTPUT
        N_m: mass distribution
    '''
        
    #A.4 from Axels diss
    lam = ( gamma( (nu+1.)/mu ) / gamma( (nu+2.)/mu ) * x )**(-mu)
    A = mu * n / (gamma( (nu + 1.)/mu )) * lam**( (nu+1.)/(mu) ) 

    #A.1 from Axels diss
    N_m	= A * x_array**nu * np.exp( -lam * x_array**mu )

    return N_m,A,lam

def bulk_velocity_pow(x,nu,mu,a_vel,b_vel,kth_moment=1):
    '''
    calculate the bulk velocity of a certain moment
    
    INPUT:
        x: mean mass
        nu,mu: shape parameters
        bvel: exponent in power law
        kth_moment: moment (0: number, 1: mass 2: reflectivity(?))
    OUTPUT:
        
    '''

    #--- apply formula 78 in Seifert+Beheng, 2006
    first_term = a_vel * gamma((kth_moment + nu + b_vel + 1.)/mu) / gamma((kth_moment + nu + 1.)/mu)
    second_term = ( gamma((nu + 1.)/mu) / gamma((nu + 2.)/mu) )**b_vel

    return first_term * second_term * x**b_vel

def main():
    #select particle
    #p_name = "column"
    p_name = "Mix2"
    #p_name = "rain_mD" #for testing m-D for rain and other properties from Mix2 #everything works fine with rain

    #some constants
    rho_w = 1000.
    N_sc = 0.710        #..Schmidt-Zahl (PK, S.541)
    n_f  = 0.333        #..Exponent von N_sc im Vent-koeff. (PK, S.541)
    nu_l = 1.50e-5

    #get class properties
    p= init_class()

    #initialize array of masses
    m_array = np.logspace(-15,-4,1000) #D-range 
    #m_array = np.linspace(1e-10,1e-2,100) #D-range ~ [6e-5m to 7e-2m]
    delta_m_array = np.diff(m_array)

    #calculate quantitites which are directly connected to particle mass
    D_array             = p[p_name].a_geo * m_array ** p[p_name].b_geo #maximum dimension
    delta_D_array       = np.diff(D_array)
    Deq_array           = ( 6. * m_array / np.pi / rho_w ) ** (1./3.) #mass equivalent diameter
    v_array_pow         = p[p_name].a_vel * m_array **p[p_name].b_vel
    v_array_Atlas       = p[p_name].a_Atlas - p[p_name].b_Atlas * np.exp( - p[p_name].c_Atlas * Deq_array)
    f_v_array_pow       = p[p_name].a_ven + p[p_name].b_ven * N_sc**n_f/nu_l**0.5 * ( v_array_pow * D_array ) ** 0.5 #single particle ventilation coefficient
    f_v_array_Atlas     = p[p_name].a_ven + p[p_name].b_ven * N_sc**n_f/nu_l**0.5 * ( v_array_Atlas * D_array ) ** 0.5 #single particle ventilation coefficient

    a_m = p[p_name].a_geo**(-1./p[p_name].b_geo) #prefactor in m=a_m D ** b
    a_star = (np.pi/6. * rho_w/a_m)**0.5
    #print(m_array,D_array,Deq_array,v_array) #uncomment to check particle properties

    #set bulk properties
    L = 1e-3 #mass concentration
    D_mean_array = np.logspace(-5,-3,50) #mean mass diameter
    xmean_array = 1./(p[p_name].a_geo)**(1./p[p_name].b_geo)*D_mean_array**(1./p[p_name].b_geo)

    ##################
    #numerical solution
    ##################

    #initialize arrays
    f_v_bulk_pow_num = np.zeros_like(D_mean_array)
    f_v_bulk_Atlas_num = np.zeros_like(D_mean_array) 
    depos_pow_num = np.zeros_like(D_mean_array) 
    depos_Atlas_num = np.zeros_like(D_mean_array) 
    N_array = np.zeros_like(D_mean_array) 
    N0D_array = np.zeros_like(D_mean_array) 
    N0m_array = np.zeros_like(D_mean_array) 

    #loop over different mean masses
    for i,Dmean in enumerate(D_mean_array):

        #get mean mass from mean mass diameter
        N_array[i] = L / xmean_array[i] #number concentration

        #get mass distribution
        N_m,N0m_array[i],lamm = generalized_gamma(N_array[i],xmean_array[i],p[p_name].nu,p[p_name].mu,m_array)
        #mue  = (p[p_name].nu+1.0)/p[p_name].b_geo - 1.0  # shape parameter of Deq distribution (assumes p[p_name].mu=1/3)
        mue  = 3.*p[p_name].nu+2  # shape parameter of Deq distribution (assumes p[p_name].mu=1/3)
        N_D,N0D_array[i],lamD = generalized_gamma(N_array[i],xmean_array[i],mue,1,D_array)

        Ntot = 0; Ltot = 0
        for i_m,m in enumerate(m_array[:-1]):
            Ntot += (N_m[i_m+1]+N_m[i_m])/2.  * delta_m_array[i_m]
            Ltot += ((N_m[i_m+1] * m_array[i_m+1] +N_m[i_m] * m_array[i_m] ))/2. * delta_m_array[i_m]
        
        print("Ntot/N_array[i],Ltot/L",Ntot/N_array[i],Ltot/L) #,Ntot,Ltot,NDtot) #check if the integration works well

        #integrate f_v 
        for i_m,m in enumerate(m_array[:-1]):
            f_v_bulk_pow_num[i] += ((D_array[i_m+1] * f_v_array_pow[i_m+1] * N_m[i_m+1]) + (D_array[i_m] * f_v_array_pow[i_m] * N_m[i_m]))/2. * delta_m_array[i_m] 
            f_v_bulk_Atlas_num[i] += ((D_array[i_m+1] * f_v_array_Atlas[i_m+1] * N_m[i_m+1]) + ( D_array[i_m] * f_v_array_Atlas[i_m] * N_m[i_m]  ))/2.  * delta_m_array[i_m] 


        depos_pow_num[i] = f_v_bulk_pow_num[i] #/N_array[i]
        depos_Atlas_num[i] = f_v_bulk_Atlas_num[i] #/N_array[i]
            
        #print( "Dmean,depos_Atlas_num[i]/depos_pow_num[i]",Dmean,depos_Atlas_num[i]/depos_pow_num[i],f_v_bulk_Atlas_numD[i]/depos_Atlas_num[i]) #display relative difference between the numeric values of the powerlaw and Atlas-type deposition (at maximum about 3%)


    ##################
    #power law analytical from Axels diss
    ##################
    a_vent_bulk = p[p_name].a_ven * gamma( (p[p_name].nu + 1 + p[p_name].b_geo) / p[p_name].mu ) / gamma((p[p_name].nu+1) / p[p_name].mu) * ( gamma((p[p_name].nu+1)/p[p_name].mu)/gamma((p[p_name].nu+2)/p[p_name].mu) )**(p[p_name].b_geo)#eq 3.121
    b_vent_bulk = p[p_name].b_ven * gamma( (p[p_name].nu+1+3./2.*p[p_name].b_geo+1./2.*p[p_name].b_vel) / p[p_name].mu ) / gamma( (p[p_name].nu+1)/ p[p_name].mu ) * ( gamma( (p[p_name].nu+1) / p[p_name].mu )/gamma( (p[p_name].nu+2)/ p[p_name].mu ) )**(3./2.*p[p_name].b_geo + 1./2. * p[p_name].b_vel)#eq 3.121

    f_vent_bulk_pow_ana = np.zeros_like(D_mean_array)
    depos_pow_ana= np.zeros_like(D_mean_array)
    for i,Dmean in enumerate(D_mean_array):
        
        #get mean mass from mean mass diameter
        x = 1./(p[p_name].a_geo)**(1./p[p_name].b_geo)*Dmean**(1./p[p_name].b_geo)

        #calculate bulk velocity
        v_bulk_pow = bulk_velocity_pow(x,p[p_name].nu,p[p_name].mu,p[p_name].a_vel,p[p_name].b_vel,kth_moment=1)
        f_vent_bulk_pow_ana[i] = a_vent_bulk + b_vent_bulk * N_sc**n_f * (v_bulk_pow * Dmean / nu_l)**0.5
        depos_pow_ana[i] = Dmean * N_array[i] * f_vent_bulk_pow_ana[i]

    ###############
    #Atlas-type analytical (based on Seifert 2014 "On the evaporation ..."
    ###############
    #initialize arrays for bulk rates
    f_vent_bulk_Atlas_ana = np.zeros_like(D_mean_array)
    f_vent_bulk_Atlas_ana2 = np.zeros_like(D_mean_array)
    depos_Atlas_ana = np.zeros_like(D_mean_array)
    depos_Atlas_ana2 = np.zeros_like(D_mean_array)

    aa = p[p_name].a_Atlas
    bb = p[p_name].b_Atlas
    cc = p[p_name].c_Atlas

    #mue  = (p[p_name].nu+1.0)/p[p_name].b_geo - 1.0  # shape parameter of Deq distribution (assumes p[p_name].mu=1/3)
    mue  = 3.*p[p_name].nu+2  # shape parameter of Deq distribution (assumes p[p_name].mu=1/3)

    #xmean_array_eq = vxmean_array * np.pi/6.*rho_w*Dmean_array
    lam_array = (np.pi/6.*rho_w*(mue+3.0)*(mue+2.0)*(mue+1.0)/xmean_array)**(1./3.)

    # chebyshev approximation of Gamma(mue+5/2)/Gamma(mue+2)
    gfak =  0.1357940435E+01 \
        + mue * ( +0.3033273220E+00  \
        + mue * ( -0.1299313363E-01  \
        + mue * ( +0.4002257774E-03  \
        - mue * 0.4856703981E-05 ) ) )
    if p_name=="rain_mD":
        mm = mue+5.0/2.0
        for i,Dmean in enumerate(D_mean_array):
            
            lam = lam_array[i]

            f_vent_bulk_Atlas_ana[i]  = p[p_name].a_ven + p[p_name].b_ven * N_sc**n_f * gfak                     \
                          * np.sqrt(aa/nu_l  / lam)                  \
                  * (1.0 - 1./2.  * (bb/aa)**1 * (lam/(1.*cc+lam))**mm \
                         - 1./8.  * (bb/aa)**2 * (lam/(2.*cc+lam))**mm \
                         - 1./16. * (bb/aa)**3 * (lam/(3.*cc+lam))**mm \
                         - 5./127.* (bb/aa)**4 * (lam/(4.*cc+lam))**mm )

            depos_Atlas_ana[i] = N_array[i] *  (mue+1) /lam * f_vent_bulk_Atlas_ana[i]
            #depos_Atlas_ana[i] = f_vent_bulk_Atlas_ana[i]*gamma(mue+2)/lam**(mue+2)

            #compare numerical to analytical solutions
            print("Dmean,pow. ana./pow. num.,Atlas ana./Atlas num.",Dmean,depos_pow_ana[i]/depos_pow_num[i],depos_Atlas_ana[i]/depos_Atlas_num[i],1./(np.pi/6.*rho_w*Dmean**3)*(0.017*Dmean**1.95))
    else: #ice (only works for m=a D**2 ?)
        mm = mue+3.
        for i,Dmean in enumerate(D_mean_array):
            
            lam = lam_array[i]

            f_vent_bulk_Atlas_ana[i]  = p[p_name].a_ven + p[p_name].b_ven * N_sc**n_f * gfak                     \
                          * np.sqrt(aa/nu_l  / lam)                  \
                  * (1.0 - 1./2.  * (bb/aa)**1 * (lam/(1.*cc+lam))**mm \
                         - 1./8.  * (bb/aa)**2 * (lam/(2.*cc+lam))**mm \
                         - 1./16. * (bb/aa)**3 * (lam/(3.*cc+lam))**mm \
                         - 5./127.* (bb/aa)**4 * (lam/(4.*cc+lam))**mm )

            depos_Atlas_ana[i] = N_array[i] * a_star *  gamma(mue+5./2.)/gamma(mue+1.) /lam**(3./2.) * f_vent_bulk_Atlas_ana[i]

            #compare numerical to analytical solutions
            print("Dmean,pow. ana./pow. num.,Atlas ana./Atlas num.",Dmean,depos_pow_ana[i]/depos_pow_num[i],depos_Atlas_ana[i]/depos_Atlas_num[i])

    #plotting
    fig,axes = plt.subplots(ncols=3)

    for i_xvar,xvar in enumerate([D_mean_array,xmean_array]):
        axes[i_xvar].semilogx(xvar,depos_pow_num/N_array,linestyle='--',color='b',label="pow. num.")
        axes[i_xvar].semilogx(xvar,depos_Atlas_num/N_array,linestyle='--',color='r',label="Atlas. num.")
        axes[i_xvar].semilogx(xvar,depos_pow_ana/N_array,linestyle='-',color='b',label="pow. ana.")
        axes[i_xvar].semilogx(xvar,depos_Atlas_ana/N_array,linestyle='-',color='r',label="Atlas ana.")

    axes[0].set_xlabel("mean mass diameter [m]")
    axes[1].set_xlabel("mean mass [kg]")
    axes[0].set_ylabel("$\int \, D \, f_v \, N(x)\, dx$")
    axes[1].set_ylabel("$\int \, D \, f_v \, N(x)\, dx$")
    plt.legend()
    #'''
    plt.show()
    '''
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    #'''

if __name__ == '__main__':

    main()
