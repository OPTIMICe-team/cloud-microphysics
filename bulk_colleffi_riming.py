'''
calculate the collision efficiency of snow-cloud droplet collisions based on the mo_check routine in McSnow
'''
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
from bulk_ventilation_coeff_SB import generalized_gamma
from pprop_dict import init_class

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx,array[idx]

def rational(x1,x2, p, q):
    """
    The general rational function description.
    p is a list with the polynomial coefficients in the numerator
    q is a list with the polynomial coefficients (except the first one)
    in the denominator
    The zeroth order coefficient of the denominator polynomial is fixed at 1.
    Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
    zeroth order denominator coefficent must comes last. (Edited.)
    """
    return np.polynomial.polynomial.polyval2d(x1,x2, p) / np.polynomial.polynomial.polyval2d(x1,x2,q)

def rational_fit_function(xdata,p0,p10,p01,p20,p02,p11,p30,p21,p12,p40,q10,q01,q20,q02,q11,q30,q21,q12,q40):
    
    #''
    #functional relationship which is adjusted in fit
    #''
    
    #rational function
    x,y   = xdata

    p   = p0 + p10 * x +  p01 * y + p20 * x**2 + p02 * y**2 + p11 * x * y + p30 * x**3 + p21 * x**2*y + p12*x*y**2 + p40 * x**4
    q   = 1.0+ q10 * x + q01 * y + q20 * x**2  + q02 * y**2 + q11 * x * y + q30 * x**3 + q21 * x**2*y + q12*x*y**2 + q40 * x**4

    return p/q 
    #return rational(x1,x2, [[p11,p12],[p21,p22]], [[1.0,q12],[q21,q22]]) #alternative: use full set of polynomial (additional parameter seem to make the fit harder)

def fit_2D_function(x,y,z): 

    from scipy import optimize

    #reshape collision efficiency array to get the right order
    zmatrix = z.reshape(len(x),len(y))
    ydata   = zmatrix.flatten(order="F")

    #transform Dsnow->x_transform and Dcloud-> y_transform
    D_snow_max      = -3.122448979591837
    x_transform     = np.log10(x) - D_snow_max
    y_transform     = y*1e5


    #ydata   = zmatrix.flatten(order="C")
    for i_cloud,Dcloud in enumerate(y):
        z_cloud     = zmatrix[:,i_cloud]
        zmax,i_zmax = [np.max(z_cloud),np.argmax(z_cloud)]
        D_snow_max  = x[i_zmax]
        print("Dcloud",Dcloud,"Dmax[mm]",10**x[i_zmax]*1e3,"log10(Dmax)",x[i_zmax],"ecoll_max",zmax)

    X, Y = np.meshgrid(x_transform,y_transform)
    xdata   = np.vstack((X.ravel(), Y.ravel()))

    x_min   = -5 - D_snow_max
    x_max   = np.log10(5e-2) - D_snow_max
    ydata   = np.ma.masked_where(xdata[0] < x_min , ydata)
    ydata   = np.ma.masked_where(xdata[0] > x_max , ydata)

    #rational_fit_function(xdata,p0,p10,p01,p20,p02,p11,p30,p21,p12,p40,q10,q01,q20,q02,q11,q30,q21,q12,q40):
    #fit_coeffs  =               [0.0,0.01,0.2,-0.2,0.0,0.0,0.0001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    fit_coeffs  =              [-0.35995764, -0.24457378,  0.78101323, -0.1019918,   0.04565127,  0.49634105,     -0.05416003,  0.,          0.59562245,  0.25315014,  1.05884148,  0.10575426,  0.31175773,  0.37308446,  0., 0., 0., 0., 0.]
    print("guess",fit_coeffs)
    fit_coeffs, covariance    = optimize.curve_fit(rational_fit_function, xdata, ydata, p0=( fit_coeffs[0],fit_coeffs[1],fit_coeffs[2],fit_coeffs[3],fit_coeffs[4],fit_coeffs[5],fit_coeffs[6],fit_coeffs[7],fit_coeffs[8],fit_coeffs[9],fit_coeffs[10],fit_coeffs[11],fit_coeffs[12],fit_coeffs[13],fit_coeffs[14],fit_coeffs[15],fit_coeffs[16],fit_coeffs[17],fit_coeffs[18]),maxfev=10000)
    print("result: fit coefficients",fit_coeffs)

    #reconstruct the fit
    fit_result  = rational_fit_function(xdata,fit_coeffs[0],fit_coeffs[1],fit_coeffs[2],fit_coeffs[3],fit_coeffs[4],fit_coeffs[5],fit_coeffs[6],fit_coeffs[7],fit_coeffs[8],fit_coeffs[9],fit_coeffs[10],fit_coeffs[11],fit_coeffs[12],fit_coeffs[13],fit_coeffs[14],fit_coeffs[15],fit_coeffs[16],fit_coeffs[17],fit_coeffs[18])
    fit_result  = fit_result.reshape(y.shape[0],x.shape[0])
    fit_result  = np.maximum(fit_result,0)
    fit_result  = np.minimum(fit_result,1)

    return fit_coeffs, fit_result

def SB06collEffi():
    '''
    reconstruct the collision efficiency in the SB scheme
    '''

    #some constants
    ecoll_min       = 0.01      #&  !..min. eff. for graupel_cloud, ice_cloud and snow_cloud
    #limits on D_c
    D_crit_c        = 10.0e-6   #, & ! D-threshold for cloud drop collection efficiency
    D_coll_c        = 40.00e-6  #! upper bound for diameter in collision efficiency
    #limits on D_s
    snow_D_crit_c   = 150.0e-6  #below that limit snow can not be rimed #150.0d-6, & !..D_crit_c
    snow_ecoll_c    = 0.80      #& !..ecoll_c

    D_c_array       = [5e-6,7e-6,10e-6,15e-6,20e-6,30e-6,40e-6,50e-6]
    e_coll_array    = np.zeros_like(D_c_array)
    for i_D_c,D_c in enumerate(D_c_array): #mean size cloud droplet

        #FORTRAN #const0 = 1.0/(D_coll_c - D_crit_c)
        const0 = 1.0/(D_coll_c - D_crit_c)
        #FORTRAN #const1 = const0 * ptype%ecoll_c
        const1 = const0 * snow_ecoll_c
        #FORTRAN #e_coll = MIN(ptype%ecoll_c, MAX(const1*(D_c - D_crit_c), ecoll_min))
        e_coll_array[i_D_c] = min(snow_ecoll_c, max(const1*(D_c - D_crit_c), ecoll_min))

    return {"D_c_array":D_c_array,"e_coll_array":e_coll_array}

def calcBulkCollEffiNumeric(df,e_coll_type="ce_boehm",k=0):
    '''
    calculate a fit to the bulk collision efficiency which was calculated numerically
    df: dataframe containing the numerical bulk collision efficiency
    e_coll_type: selection of particle-particle collision efficieny (so far here only ce_boehm is implemented, but extension should be easy)
    k: moment of the PSD
    '''

    #process dataframe which contains particle-particle collision efficiencies
    dfD_clouddrop_array   = np.array(df["D_clouddrop"].drop_duplicates()) #get all different D_clouddrop values
    dfD_clouddrop_array   = np.array(dfD_clouddrop_array[~np.isnan(dfD_clouddrop_array)]) #remove nans from list
    dfD_snow_array        = df["D_snow"].drop_duplicates() #get all different D_clouddrop values
    dfD_snow_array        = np.array(dfD_snow_array[~np.isnan(dfD_snow_array)]) #remove nans from list

    #create a matrix of ce_boehm with dimensions [D_clouddrop,D_snow]
    df_ce_boehm_array = np.zeros([dfD_clouddrop_array.shape[0],dfD_snow_array.shape[0]])
    for i_Dc,D_clouddrop in enumerate(dfD_clouddrop_array):
        df_thisDc = df.loc[df["D_clouddrop"]==D_clouddrop] #select subset of Dataframe
        df_ce_boehm_array[i_Dc,:]   = df_thisDc["ce_boehm"]
    
    #get class properties
    p= init_class()
    ps_name  = "Mix2"
    pc_name = "cloud_nuemue1"

    #initialize array of masses
    m_snow_array    = np.logspace(-14,-3,100)
    m_cloud_array   = np.logspace(-18,-8,100)

    delta_m_snow_array  = np.diff(m_snow_array)
    delta_m_cloud_array = np.diff(m_cloud_array)

    #calculate quantitites which are directly connected to particle mass
    D_array_snow             = p[ps_name].a_geo * m_snow_array ** p[ps_name].b_geo #maximum dimension
    D_array_cloud            = p[pc_name].a_geo * m_cloud_array ** p[pc_name].b_geo #maximum dimension

    #set range for evaluating the bulk coll. efficiency
    D_mean_snow_array   = np.logspace(-5,-1,50) #mean mass diameter
    D_mean_cloud_array  = np.array([5e-6,7e-6,10e-6,15e-6,20e-6,30e-6,50e-6])[::-1] #mean mass diameter
    xmean_snow_array = 1./(p[ps_name].a_geo)**(1./p[ps_name].b_geo)*D_mean_snow_array**(1./p[ps_name].b_geo)
    xmean_cloud_array = 1./(p[pc_name].a_geo)**(1./p[pc_name].b_geo)*D_mean_cloud_array**(1./p[pc_name].b_geo)

    #set bulk properties
    L = 1e-5 #mass concentration

    #initialize arrays
    Nsnow_array             = np.zeros_like(D_mean_snow_array) 
    Ncloud_array            = np.zeros_like(D_mean_cloud_array) 
    bulkCollEffi_array      = np.zeros([D_mean_snow_array.shape[0],D_mean_cloud_array.shape[0]])    
    N0m_array_snow          = np.zeros_like(D_mean_snow_array)     
    N0m_array_cloud         = np.zeros_like(D_mean_cloud_array)     
    #loop over different mean masses
    for i_s,Dmean_snow in enumerate(D_mean_snow_array):
        #get mean mass from mean mass di_sameter
        Nsnow_array[i_s] = L / xmean_snow_array[i_s] #number concentration

        #get mass di_sstri_sbuti_son
        N_m_snow,N0m_array_snow[i_s],lamm = generalized_gamma(Nsnow_array[i_s],xmean_snow_array[i_s],p[ps_name].nu_SB,p[ps_name].mu_SB,m_snow_array)

        Ntot = 0; Ltot = 0
        for i_ms,m in enumerate(m_snow_array[:-1]):
            Ntot += (N_m_snow[i_ms+1]+N_m_snow[i_ms])/2.  * delta_m_snow_array[i_ms]
            Ltot += ((N_m_snow[i_ms+1] * m_snow_array[i_ms+1] +N_m_snow[i_ms] * m_snow_array[i_ms] ))/2. * delta_m_snow_array[i_ms]
            
        print("D_mean_snow, Ntot/N_array[i],Ltot/L",Dmean_snow,Ntot/Nsnow_array[i_s],Ltot/L) #check if the integration works well
        for i_c,Dmean_cloud in enumerate(D_mean_cloud_array):
            #get mean mass from mean mass diameter
            Ncloud_array[i_c] = L / xmean_cloud_array[i_c] #number concentratiion
            #get mass di_cstri_cbuti_con
            N_m_cloud,N0m_array_cloud[i_c],lamm = generalized_gamma(Ncloud_array[i_c],xmean_cloud_array[i_c],p[pc_name].nu_SB,p[pc_name].mu_SB,m_cloud_array)

            Ntot = 0; Ltot = 0
            for i_mc,m in enumerate(m_cloud_array[:-1]):
                Ntot += (N_m_cloud[i_mc+1]+N_m_cloud[i_mc])/2.  * delta_m_cloud_array[i_mc]
                Ltot += ((N_m_cloud[i_mc+1] * m_cloud_array[i_mc+1] +N_m_cloud[i_mc] * m_cloud_array[i_mc] ))/2. * delta_m_cloud_array[i_mc]
            if i_s==0: #it is enough to check the integration once
                print("D_mean_cloud, Ntot/N_array[i],Ltot/L",Dmean_cloud,Ntot/Ncloud_array[i_c],Ltot/L)  #check if the integration works well

            #integrate e_coll
            normalization   = 0
            for i_ms,m in enumerate(m_snow_array[:-1]):
                for i_mc,m in enumerate(m_cloud_array[:-1]):
                    i_snow_nearest,dummy    = find_nearest(dfD_snow_array, D_array_snow[i_ms+1])
                    i_cloud_nearest,dummy    = find_nearest(dfD_clouddrop_array, D_array_cloud[i_mc+1])
                    collEff                 =  df_ce_boehm_array[i_cloud_nearest,i_snow_nearest]  #get the collision efficiency from the nearest datapoint (simple search along Dclouddrop and Dsnow
                    normalization                 += ((D_array_snow[i_ms+1]**2 * D_array_cloud[i_mc]**2 * N_m_snow[i_ms+1] * N_m_cloud[i_mc+1] * m_cloud_array[i_mc+1]**k) + (D_array_snow[i_ms]**2 * D_array_cloud[i_mc]**2 * N_m_snow[i_ms] * N_m_cloud[i_mc] * m_cloud_array[i_mc]**k))/2. * delta_m_snow_array[i_ms] * delta_m_cloud_array[i_mc] 
                    bulkCollEffi_array[i_s,i_c]   +=  ((collEff * D_array_snow[i_ms+1]**2 * D_array_cloud[i_mc]**2 * N_m_snow[i_ms+1] * N_m_cloud[i_mc+1] * m_cloud_array[i_mc+1]**k) + (collEff * D_array_snow[i_ms]**2 * D_array_cloud[i_mc]**2 * N_m_snow[i_ms] * N_m_cloud[i_mc] * m_cloud_array[i_mc]**k))/2. * delta_m_snow_array[i_ms] * delta_m_cloud_array[i_mc] 
            bulkCollEffi_array[i_s,i_c]     = bulkCollEffi_array[i_s,i_c] / normalization

    bulkCollEffi_dic    = dict()
    bulkCollEffi_dic["D_mean_snow_array"]   = D_mean_snow_array
    bulkCollEffi_dic["D_mean_cloud_array"]   = D_mean_cloud_array
    bulkCollEffi_dic["ce_boehm_bulk"]       = bulkCollEffi_array
    bulkCollEffi_dic["moment"]       = k

    return bulkCollEffi_dic

def readPartPartCollEffi(vterm="3",rimegeo="3"):
    '''
    read the particle based collision efficiency as calculated from the mo_check routine in McSnow
    INPUT:
        vterm: hydrodynamic model (1: HW10, 2: KC05, 3: bohm) 
        rimegeo: geometry of rimed particles (1: bulk, 2: fillin, 3: similarity)
    OUTPUT:
        panda dataframe containing
    '''
    colnames=['D_snow','D_clouddrop', 'ce_boehm', 'ce_bulkcoberlist','ce_spheresHallBeheng',"rho_rime","vel_relative","StokesNumber","N_Reynolds","N_ReBig","ce_BeardGrover","ce_Robin","ce_Slinn","ce_Holger"] 
    df = pd.read_csv("/home/mkarrer/Dokumente/McSnow_checks/experiments/check_c" + rimegeo + "_v" + vterm + "/colleff_riming_Fr000_rhor600.dat",names=colnames,header=None,delimiter=" ")    

    return df

def plot_colleffi(axes,df,SB06collEffi_dic,BulkCollEffi_dic):
    '''
    plot the particle-particle and the bulk collision efficiencies
    INPUT:
        axes:   axes handle
        df:     pandas Dataframe containing the particle-particle collision efficiency        
        SB06collEffi_dic:   dictionary containing the SB06 implementation evaluated along the Dsnow and Dcloud arrays
        BulkCollEffi_dic:   dictionary containing the numerical integration of e_coll_bulk along the Dsnow and Dcloud arrays
    OUTPUT:
        axes:   axes handle
    '''

    ###plot Ecoll against Dsnow with colored lines of Dcloud
    #select array of Dcloud sizes
    D_clouddrop_array   = df["D_clouddrop"].drop_duplicates() #get all different D_clouddrop values
    D_clouddrop_array   = D_clouddrop_array[~np.isnan(D_clouddrop_array)] #remove nans from list
    D_clouddrop_array   = D_clouddrop_array[::-1] #invert axis to start with largest values
    lines_partDc        = []
    for i_Dc,D_clouddrop in enumerate(D_clouddrop_array):
        df_thisDc = df.loc[df["D_clouddrop"]==D_clouddrop] #select subset of Dataframe
        l = axes[0][0].semilogx(df_thisDc["D_snow"]*1e3,df_thisDc["ce_boehm"],label="D$_{cloud}$=" + "{:.0f}".format(D_clouddrop*1e6) + "$\mu$ m" )
        lines_partDc.append(l)

    #set labels and limits
    axes[0][0].set_xlabel("D$_{snow}$ [mm]")
    axes[0][0].set_ylabel("E$_{coll}$")
    axes[0][0].set_xlim([1e-2,5e1])
    axes[0][0].set_ylim([0,1])
    axes[0][0].grid(which="major",b=True)
    axes[0][0].legend()

    ###plot Ecoll against Dcloud with colored lines of Dsnow
    #select array of snow particle sizes
    D_snow_array        = df["D_snow"].drop_duplicates() #get all different D_clouddrop values
    D_snow_array        = np.array(D_snow_array[~np.isnan(D_snow_array)]) #remove nans from list
    for i_Ds,D_snow in enumerate(D_snow_array[50:150:20]):
        df_thisDs = df.loc[df["D_snow"]==D_snow] #select subset of Dataframe
        axes[0][1].plot(df_thisDs["D_clouddrop"]*1e6,df_thisDs["ce_boehm"],label="D$_{snow}$=" + "{:.3f}".format(D_snow*1e3) + "mm" )
        axes[0][1].plot(df_thisDs["D_clouddrop"]*1e6,df_thisDs["ce_boehm"],label="D$_{snow}$=" + "{:.3f}".format(D_snow*1e3) + "mm" )

    #set labels and limits
    axes[0][1].set_xlabel("D$_{cloud}$ [$\mu$m]")
    axes[0][1].set_ylabel("E$_{coll}$")
    axes[0][1].set_xlim([5,50])
    axes[0][1].set_ylim([0,1])
    axes[0][1].grid(which="both",b=True)
    axes[0][1].legend()
    
    ###plot Ecoll against Dsnow as implemented in SB06 and calculated numerically (+fits)
    D_snow  = np.logspace(-3,3,100)
    for i_Dc,D_clouddrop in enumerate(D_clouddrop_array):
        #differentiate Sb06 and numerical in legend
        if i_Dc==0:
            axes[1][0].semilogx(np.nan,np.nan,label="SB06",color="k",linestyle="--")
            axes[1][0].semilogx(np.nan,np.nan,label="numerical ("+ str(BulkCollEffi_dic["moment"]) + "th mom.)",color="k",linestyle="-")
            axes[1][0].semilogx(np.nan,np.nan,label="fit to numerical",color="k",linestyle=":")

        i_Dc_dic    = np.where(np.array(SB06collEffi_dic["D_c_array"])==D_clouddrop)[0][0] #find the corresponding value in the dictionary
        e_coll  = np.where(D_snow>0.15,SB06collEffi_dic["e_coll_array"][i_Dc_dic],0.0) #apply: snow_D_crit_c   
        l   = axes[1][0].semilogx(D_snow,e_coll,label="__SB06 D$_{cloud}$=" + "{:.0f}".format(D_clouddrop*1e6) + "$\mu$m",linestyle="--") #,color=lines_partDc[i_Dc][0].get_color())

        if not all(np.array(D_clouddrop_array)==BulkCollEffi_dic["D_mean_cloud_array"]):
            print("D_clouddrop_array and BulkCollEffi_dic[D_mean_cloud_array] are not identical: please check")

        axes[1][0].semilogx(BulkCollEffi_dic["D_mean_snow_array"]*1e3,BulkCollEffi_dic["ce_boehm_bulk"][:,i_Dc],label="D$_{cloud}$=" + "{:.0f}".format(D_clouddrop*1e6) + "$\mu$m",linestyle="-",color=l[0].get_color())
        axes[1][0].semilogx(BulkCollEffi_dic["D_mean_snow_array"]*1e3,BulkCollEffi_dic["ce_boehm_bulk_fitted"][i_Dc,:],label="__D$_{cloud}$=" + "{:.0f}".format(D_clouddrop*1e6) + "$\mu$m",linestyle=":",color=l[0].get_color())
        


    #set labels and limits
    axes[1][0].set_xscale("log")
    axes[1][0].set_xlabel("D$_{snow}$ [mm]")
    axes[1][0].set_ylabel("E$_{coll,bulk}$")
    axes[1][0].set_xlim([1e-2,5e1])
    axes[1][0].set_ylim([0,1])
    axes[1][0].grid(which="major",b=True)
    axes[1][0].legend()

    ###plot Ecoll against Dcloud as imlemented in SB06
    axes[1][1].plot(np.array(SB06collEffi_dic["D_c_array"])*1e6,SB06collEffi_dic["e_coll_array"],label="SB06 D$_{snow}$>150$\mu$m",linestyle="--")
    axes[1][1].plot(np.nan,np.nan,label="numerical ("+ str(BulkCollEffi_dic["moment"]) + "th mom.)",color="k",linestyle="None") #for legend only
    for i_Ds,D_snow in enumerate(BulkCollEffi_dic["D_mean_snow_array"]):
        if i_Ds%3 != 0: #plot only every 5th but dont mess with the index
            continue
        axes[1][1].plot(BulkCollEffi_dic["D_mean_cloud_array"]*1e6,BulkCollEffi_dic["ce_boehm_bulk"][i_Ds,:],label="D$_{snow}$=" + "{:.3f}".format(D_snow*1e3) + "mm",linestyle="-")

    #set labels and limits
    axes[1][1].set_xlabel("D$_{cloud}$ [$\mu$m]")
    axes[1][1].set_ylabel("E$_{coll,bulk}$")
    axes[1][1].set_xlim([5,50])
    axes[1][1].set_ylim([0,1])
    axes[1][1].grid(which="both",b=True)
    axes[1][1].legend()

    return axes

if __name__ == '__main__':
    import argparse
    import pickle
    parser =  argparse.ArgumentParser(description='plot particle-based collision efficiency and ')

    parser.add_argument('-col','--calc_or_load', nargs=1, help='calculate the bulk collision efficiencies (1) or load them from previous calculation (0)')

    parser.add_argument('-m','--moment', nargs=1, help='which moment to use for the bulk calculations?')

    args = parser.parse_args()
    calc_or_load = (args.calc_or_load[0]=="1")
    moment = int(args.moment[0])

    #set up figure
    fig,axes    = plt.subplots(nrows=2,ncols=2,figsize=(12,12))   

    #get data from McSnow's check routine
    df      = readPartPartCollEffi(vterm="1",rimegeo="3")

    #get parameterization from SB06    
    SB06collEffi_dic    = SB06collEffi()
    if calc_or_load: 
        
        #calculate the bulk collision efficiency numerically 
        BulkCollEffi_dic    = calcBulkCollEffiNumeric(df,e_coll_type="ce_boehm",k=moment)

        with open('BulkEc'+ str(BulkCollEffi_dic["moment"]) +  '.pickle', 'wb') as handle:
            pickle.dump(BulkCollEffi_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('BulkEc'+ str(moment) +  '.pickle', 'rb') as handle:
            BulkCollEffi_dic = pickle.load(handle)

    #make a fit to the bulk collision efficiency
    #BulkCollEffi_dic    = calcBulkFit(BulkCollEffi_dic)
    fitcoeffs,BulkCollEffi_dic["ce_boehm_bulk_fitted"] = fit_2D_function(BulkCollEffi_dic['D_mean_snow_array'].flatten(),BulkCollEffi_dic['D_mean_cloud_array'].flatten(),BulkCollEffi_dic['ce_boehm_bulk'].flatten()) 

    #plot all coll. efficiencies
    axes    = plot_colleffi(axes,df,SB06collEffi_dic,BulkCollEffi_dic)

    plt.savefig("colleffi_mom"+ str(BulkCollEffi_dic["moment"]) + ".png")
