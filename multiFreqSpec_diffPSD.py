'''
visualize the particle size (or mass) distribution and Doppler spectrum for different widths of the particle mass distribution (PSM)
'''

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

import sys
sys.path.append("/home/mkarrer/Dokumente/snowScatt/examples/GMD_paper_2021") #change to your path of snowscatt
import snowScatt #if this fails use python3 !!
from snowScatt.instrumentSimulator.radarMoments import Ze
from snowScatt.instrumentSimulator.radarMoments import calcMoments
from snowScatt.instrumentSimulator.radarSpectrum import dopplerSpectrum
from snowScatt.instrumentSimulator.PSD import GammaPSD


def Nexp(D, lam):
    return np.exp(-lam*D)


def dB(x):
    return 10.0*np.log10(x)

def plot_PSDandSpec(ax1,ax2,ax3,ax4,ax5,ax6,D0=1e-2,Nw=1e4,mu_Dmax=None,linestyle="-",label="label not set",particle="vonTerzi_mixcoldend" ):

    '''
    plot the spectrum for given hydrometeor moments (q and N) and nu-parameter in N(m)=N_0*m**nu_SB*exp(-lam*m**(1./3.))
    D0: the median volume diameter.

    Nw: the intercept parameter (shouldnt matter for the shape of the spectrum)
    mu_Dmax: mu-parameter in N(D)=N0 D^mu exp(-lam D)
    ax1: plot the particle mass distribution
    ax2: plot the Doppler spectrum
    label: label appears in legend
    particle: particle as named in snowscatt
    '''


    #calculate the particle mass distribution
    Dmax = np.linspace(0.3e-3, 20.0e-3, 1000) # list of sizes
    #Dmax = np.logspace(-5, -1, 100) # list of sizes
    concentration = 1000.0 # number m**-1 m**-3
 
    PSD = GammaPSD(D0=D0, Nw=Nw, mu=mu_Dmax,D_max=1e-2)
    N_D = concentration*PSD(Dmax)

    #plot the P  nspace(0.1e-3, 20.0e-3, 1000) # list of sizes
    ax1.loglog(Dmax,N_D,linestyle=linestyle,label=label,color="k")

    ax1.set_xlabel("D [m]")
    ax1.set_ylabel("f(D) [m-4]")
    
    ax1.set_xlim(3e-4,1e-2)
    ax1.set_ylim(1e-5,5e-2)

    #calculate Ze and DWR


    frequency =  np.array([13.6e9, 35.6e9, 94.0e9])
    freq_label =  ['X-band', 'Ka-band', 'W-band'] # frequencies
    wl  = np.zeros_like(frequency)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    spec = [None,None,None]
    spec_dB = [None,None,None]
    for idf,freq in enumerate(frequency):

        #calculate wavelength
        wl[idf] = snowScatt._compute._c/freq

        #calculate spectrum
        spec[idf],vel   = dopplerSpectrum(Dmax, np.array([N_D,]), wl[idf], particle, temperature=270)
        spec_dB[idf]    = dB(spec[idf]*np.gradient(vel))

        #calculate and display some moments of the PSD
        Ze,MDV,SW,SK = calcMoments(spec[idf][0,:], vel, n=4)
        Ze = dB(Ze) #convert to dBz
        print("mu",mu_Dmax,freq_label[idf],"Ze [dBz]",Ze,"MDV [m/s]",MDV)

        #plot the spectrum
        if linestyle=="-":
            label = freq_label[idf]
        else:
            label = "__None"
        ax2.plot(vel, spec_dB[idf][0, :],label=label,linestyle=linestyle,color=colors[idf])

    ###plot the dual spectrum ratios
    #X-Ka
    ax3.plot(vel,spec_dB[0][0,:]-spec_dB[1][0,:],linestyle=linestyle,color="k")
    ax4.plot(vel,spec_dB[1][0,:]-spec_dB[2][0,:],linestyle=linestyle,color="k")

    ###plot the dual spectrum ratios against each other
    #X-Ka
    ax5.plot(spec_dB[1][0,:]-spec_dB[2][0,:],spec_dB[0][0,:]-spec_dB[1][0,:],linestyle=linestyle,color="k")
    ax5.set_xlabel('DSR$_{Ka,W}$ [dB]')
    ax5.set_ylabel('DSR$_{X,Ka}$ [dB]')

    for ax in [ax3,ax4]:
        ax.set_xlabel('velocity [m/s]')
        ax.set_ylim([-1,15])
    ax3.set_ylabel('DSR$_{X,Ka}$ [dB]')
    ax4.set_ylabel('DSR$_{Ka,W}$ [dB]')

    for ax in [ax1,ax2]:
        ax.grid(b=True,which="both")
    ax2.set_ylabel('spectral power')
    ax2.set_ylim([-100, -70])
    ax2.legend()
    ax2.set_xlabel('velocity [m/s]')

    return ax1,ax,ax3,ax4,ax5,ax6

if __name__ == "__main__":

    #initialize mass array and moments
    D0           = 5e-3
    Nw           = 1e-4

    particle="vonTerzi_mixcoldend" #select particle properties from snowscatt

    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6))      = plt.subplots(nrows=3,ncols=2,figsize=(12,12))

    #different nu_parameter
    mu_Dmax1 = 1.0
    mu_Dmax2 = 5.0
    ax1,ax2,ax3,ax4,ax5,ax6      = plot_PSDandSpec(ax1,ax2,ax3,ax4,ax5,ax6,D0=D0,Nw=Nw,mu_Dmax=mu_Dmax1,label=r"$\mu_{Dmax}$ = " + str(mu_Dmax1),linestyle="-",particle=particle)
    ax1,ax2,ax3,ax4,ax5,ax6      = plot_PSDandSpec(ax1,ax2,ax3,ax4,ax5,ax6,D0=D0,Nw=Nw,mu_Dmax=mu_Dmax2,label=r"$\mu_{Dmax}$ = "+ str(mu_Dmax2),linestyle="--",particle=particle)
    ax1.legend()

    plt.tight_layout()
    savepath    = "plots/"
    filename    = "MultiFreqSpec_diffPSD" 
    fig.tight_layout()
    plt.savefig(savepath + filename + ".png")
    #plt.savefig(savepath + filename + ".pdf")

