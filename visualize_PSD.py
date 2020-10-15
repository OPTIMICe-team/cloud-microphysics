'''
visualize the particle size (or mass) distribution for some cases
'''
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

from pprop_dict import init_class,main,calculate_PSM,calculate_moments_and_fluxes_from_PSM


def plot_PSM(ax,q,N,mu_SB=None,linestyle="-",label="reference"):

    if not mu_SB is None: #if None take default read from pprop_dict
        curr_cat.mu_SB = mu_SB
    f_m         = calculate_PSM(q,N,curr_cat,mass_array)
    moments     = calculate_moments_and_fluxes_from_PSM(mass_array,f_m,curr_cat)
    m_mean      = moments[1]/moments[0]

    ax.loglog(mass_array,f_m,linestyle=linestyle,label=label,color="k")
    ax.axvline(x=m_mean,color="k",linestyle=linestyle)

    ax.set_xlabel("mass m [kg]")
    ax.set_ylabel("f(m)")
    
    ax.set_xlim(mass_array[0],mass_array[-1])
    ax.set_ylim(1e-8,1e13)

    return ax,moments

def add_string(ax,mom1,mom2):

    ax.legend(loc="lower left")
    str_an = r"M(0)={:.1e}$\rightarrow${:.1e}".format(mom1[0],mom2[0])
    str_an += "\n"
    str_an += r"M(1)={:.1e}$\rightarrow${:.1e}".format(mom1[1],mom2[1])
    str_an += "\n"
    str_an += r"M(2)={:.1e}$\rightarrow${:.1e}".format(mom1[2],mom2[2])
    ax.text(0.99, 0.99, str_an,
     horizontalalignment='right',
     verticalalignment='top',
     transform = ax.transAxes)

    return ax

if __name__ == "__main__":
    p= init_class()

    #initialize mass array and moments
    mass_array  = np.logspace(-10,-3,100)
    q           = 1e-4
    N           = 1e3

    curr_cat    = p["Mix2"]

    fig,axes      = plt.subplots(nrows=2,ncols=2,figsize=(12,12))

    #mass doubling
    axes[0,0],mom1      = plot_PSM(axes[0,0],q,N)
    axes[0,0],mom2      = plot_PSM(axes[0,0],2*q,N,label="mass * 2",linestyle="--")
    axes[0,0]           = add_string(axes[0,0],mom1,mom2)

    #number doubling
    axes[0,1],mom1      = plot_PSM(axes[0,1],q,N)
    axes[0,1],mom2      = plot_PSM(axes[0,1],q,0.5*N,label="number * 0.5",linestyle="--")
    axes[0,1]           = add_string(axes[0,1],mom1,mom2)

    #spectrum broadening
    axes[1,0],mom1      = plot_PSM(axes[1,0],q,N)
    axes[1,0],mom2      = plot_PSM(axes[1,0],q,N,mu_SB=1.0,label=r"mu(m) 0 $\rightarrow$ 1",linestyle="--")
    axes[1,0]           = add_string(axes[1,0],mom1,mom2)

    #change of q and N simultaneously
    axes[1,1],mom1      = plot_PSM(axes[1,1],q,N)
    #for qfac in np.linspace(0,1.0,100): #figure out which factor we need to have a constant M(2)
    #    axes[1,1],mom2      = plot_PSM(axes[1,1],qfac*q,0.5*N,label="M(0) and M(1) changes\n but M(2) is constant",linestyle="--")
    #    print(qfac,mom2[2]/mom1[2])
    axes[1,1],mom2      = plot_PSM(axes[1,1],0.708*q,0.5*N,label="M(0) and M(1) changes\n but M(2) is constant",linestyle="--")
    
    axes[1,1]           = add_string(axes[1,1],mom1,mom2)

    plt.savefig("/home/mkarrer/Dokumente/plots/distributions/PSD_examples.png")
    print("png is at plots/distributions/PSD_examples.png")

