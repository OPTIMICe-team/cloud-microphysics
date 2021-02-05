'''
plot collision rates from collint
'''
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

def read_coll_rates(pairs):
    import pandas as pd


    path    = "/home/mkarrer/Dokumente/Seifert2014_gitlab/collint/DATA/"

    if pairs=="ice_ice":
        colnames    = ["L_i","N_i","D_i","v_i","mue","coll_n_ana", "coll_n_num", "coll_n_num2", "coll_n_sbb", "coll_n_rog","coll_q_ana", "coll_q_num", "coll_q_num2", "coll_q_sbb", "coll_q_rog"]
        filename    = "collint_ice_ice_10.dat" #so far only m=1.0 is read in
    
    elif pairs=="snow_snow":
        colnames    = ["L_s","N_s","D_s","v_s","mue","coll_n_ana", "coll_n_num", "coll_n_num2", "coll_n_sbb", "coll_n_rog"]
        filename    = "collint_snow_snow_10.dat" #so far only m=1.0 is read in
    elif pairs=="ice_snow":
        colnames    = ["L_s","N_s","D_r","v_s","L_i","N_i","D_i","v_i",
                    "coll_n_ana", "coll_n_num", "coll_n_wrg",
                    "coll_q_ana", "coll_q_num", "coll_q_wrg",
                    "coll_n_wis", "coll_q_wis",                                  
                     "coll_n_miz", "coll_q_miz",
                     "coll_n_rog", "coll_q_rog",
                     "coll_n_num2","coll_q_num2",
                     "coll_n_sbb", "coll_q_sbb",
                     "coll_n_rog_D", "coll_q_rog_D",
                     "D_s","D_eq_i"] 
        filename    = "collint_ice_snow_10.dat" #so far only m=1.0 is read in

    df = pd.read_csv(path + filename,names=colnames,header=None,delimiter="\s+")

    return df

def plot_coll_rates_selfcollection(axes,df,pairs):

    if pairs=="ice_ice":
        mue_array   = [2,8] #(3.0*nu_mass+2.0): nu_mass=0 is mue=2; nu_mass=2 is mue=8
        Dvar        = "D_i"
        D_var_str   = "D$_{max,ice}$"
        moments_short   = ["n","q"]
        
    if pairs=="snow_snow":
        mue_array   = [2,8] #(3.0*nu_mass+2.0): nu_mass=0 is mue=2; nu_mass=2 is mue=8
        Dvar        = "D_s"
        D_var_str   = "D$_{max,snow}$"
        moments_short   = ["n"]
  
    colors       = ["blue","green","yellow"] 
    colors2      = ["red","orange","black"] 
    for i_mue,mue in enumerate(mue_array):
        df_this_mue = df.loc[df["mue"]==mue]

        for i_moment,moment_short in enumerate(moments_short):
            #D-kernel
            axes[0][i_moment].loglog(df_this_mue[Dvar]*1e3,df_this_mue["coll_" + moment_short + "_num2"],linestyle="-",color=colors[i_mue],label="_None") #numeric
            axes[0][i_moment].loglog(df_this_mue[Dvar]*1e3,df_this_mue["coll_" + moment_short + "_ana"],linestyle=":",color=colors[i_mue],alpha=0.4,label="_None") #numeric
            axes[0][i_moment].loglog(df_this_mue[Dvar]*1e3,df_this_mue["coll_" + moment_short + "_sbb"],linestyle="--",color=colors[i_mue],label="_None") #analytic

            #A-kernel
            axes[1][i_moment].loglog(df_this_mue[Dvar]*1e3,df_this_mue["coll_" + moment_short + "_num"],linestyle="-",color=colors[i_mue],label="_None") #numeric
            axes[1][i_moment].loglog(df_this_mue[Dvar]*1e3,df_this_mue["coll_" + moment_short + "_rog"],linestyle="--",color=colors[i_mue],label="_None") #numeric

    #add labels
    for ax in axes.flatten(): 
        for i_mue,mue in enumerate(mue_array):
            ax.plot(np.nan,np.nan,color=colors[i_mue],label="$\mu_{eq}$=" + "{:.0f}".format(mue))
    for ax in axes[0].flatten(): 
        ax.plot(np.nan,np.nan,linestyle="-",label="numeric",color="k")
        ax.plot(np.nan,np.nan,linestyle="--",label="analytic",color="k")
        ax.plot(np.nan,np.nan,linestyle=":",label="old analytic",alpha=0.4,color="k")
    for ax in axes[1].flatten(): 
        ax.plot(np.nan,np.nan,linestyle="-",label="numeric",color="k")
        ax.plot(np.nan,np.nan,linestyle="--",label="analytic",color="k")
           
 
    for ax in axes.flatten(): 
        ax.set_xlim([1e-2,1e1])
        ax.set_xlabel(D_var_str + " [mm]")
        ax.legend()
    axes[0][0].set_ylim([2e-2,2e0]) #number collision rate
    axes[1][0].set_ylim([2e-2,2e0]) #number collision rate
    if pairs in ["ice_ice"]: #snow_selfcollection has no mass collision rate
        axes[0][1].set_ylim([3e-2,2e1]) #mass collision rate
        axes[1][1].set_ylim([3e-2,2e1]) #mass collision rate

    for i in [0,1]:
        axes[i][0].set_ylabel("normalized number collision rate [m/s]")
        if pairs in ["ice_ice"]: #snow_selfcollection has no mass collision rate
            axes[i][1].set_ylabel("normalized mass collision rate [m/s]")

    for i in [0,1]:
        if pairs in ["snow_snow"] and i>0: #snow_selfcollection has no mass collision rate
            continue
        axes[0][i].set_title("D-kernel")
        axes[1][i].set_title("A-kernel")

    plt.tight_layout()
    plt.savefig("collint_plots/" + pairs + ".pdf") 
    print("saved: collint_plots/" + pairs + ".pdf") 

def plot_coll_rates_icesnowcollection(axes,df,pairs):

    #(3.0*nu_mass+2.0): nu_mass=0 is mue=2; nu_mass=2 is mue=8;check how nu_mass and mue is set for the two categories in the collint.f90 code
        
    Dvar        = "D_s"
    D_var_str   = "D$_{max,snow}$"
    moments_short   = ["n","q"]
  
    colors      = ["red","orange","blue","green",""] 
    D_i_array   = [0.2,0.1,0.05,0.01]

    for i_D_i,D_i in enumerate(D_i_array):
        df_this_Di = df.loc[(df["D_i"]*1e3)==D_i]

        for i_moment,moment_short in enumerate(moments_short):
            #D-kernel
            axes[0][i_moment].loglog(df_this_Di[Dvar]*1e3,df_this_Di["coll_" + moment_short + "_num2"],linestyle="-",color=colors[i_D_i],label="_None") #numeric
            axes[0][i_moment].loglog(df_this_Di[Dvar]*1e3,df_this_Di["coll_" + moment_short + "_ana"],linestyle=":",color=colors[i_D_i],alpha=0.4,label="_None") #numeric
            axes[0][i_moment].loglog(df_this_Di[Dvar]*1e3,df_this_Di["coll_" + moment_short + "_rog_D"],linestyle="--",color=colors[i_D_i],label="_None") #analytic

            #A-kernel
            axes[1][i_moment].loglog(df_this_Di[Dvar]*1e3,df_this_Di["coll_" + moment_short + "_num"],linestyle="-",color=colors[i_D_i],label="_None") #numeric
            axes[1][i_moment].loglog(df_this_Di[Dvar]*1e3,df_this_Di["coll_" + moment_short + "_rog"],linestyle="--",color=colors[i_D_i],label="_None") #numeric

    #add labels
    for ax in axes.flatten(): 
        for i_D_i,D_i in enumerate(D_i_array):
            ax.plot(np.nan,np.nan,color=colors[i_D_i],label="$D_{max,i}$=" + "{:.2f}".format(D_i)+ "mm")
    for ax in axes[0].flatten(): 
        ax.plot(np.nan,np.nan,linestyle="-",label="numeric",color="k")
        ax.plot(np.nan,np.nan,linestyle="--",label="analytic",color="k")
        ax.plot(np.nan,np.nan,linestyle=":",label="old analytic",alpha=0.4,color="k")
    for ax in axes[1].flatten(): 
        ax.plot(np.nan,np.nan,linestyle="-",label="numeric",color="k")
        ax.plot(np.nan,np.nan,linestyle="--",label="analytic",color="k")
           
 
    for ax in axes.flatten(): 
        #ax.set_xlim([1e-2,3e0])
        ax.set_xlim([5e-3,10e0])
        ax.set_xlabel(D_var_str + " [mm]")
        ax.legend()
    axes[0][0].set_ylim([1e-2,1e0]) #number collision rate
    axes[1][0].set_ylim([1e-2,1e0]) #number collision rate
    axes[0][1].set_ylim([3e-2,5e0]) #mass collision rate
    axes[1][1].set_ylim([3e-2,5e0]) #mass collision rate

    for i in [0,1]:
        axes[i][0].set_ylabel("normalized number collision rate [m/s]")
        axes[i][1].set_ylabel("normalized mass collision rate [m/s]")

    for i in [0,1]:
        axes[0][i].set_title("D-kernel")
        axes[1][i].set_title("A-kernel")

    plt.tight_layout()
    plt.savefig("collint_plots/" + pairs + ".pdf") 
    print("saved: collint_plots/" + pairs + ".pdf") 

if __name__ == '__main__':
    import argparse
    import pickle
    parser =  argparse.ArgumentParser(description='plot collision rates from collint')

    parser.add_argument('-p','--pairs', nargs=1, help='which collision pairs should be plotted? so far available (ice_ice,ice_snow,snow_snow)')

    args = parser.parse_args()
    pairs = args.pairs[0]

    #set up figure
    if pairs in ["ice_ice"]:
        fig,axes    = plt.subplots(nrows=2,ncols=2,figsize=(7,7))   
    elif pairs in ["ice_snow"]:
        fig,axes    = plt.subplots(nrows=2,ncols=2,figsize=(9,9))   
    elif pairs in ["snow_snow"]:
        fig,axes    = plt.subplots(nrows=2,ncols=1,figsize=(3.5,7))   
        axes        = axes[...,np.newaxis]

    #read collint output
    df  = read_coll_rates(pairs)
    
    #plot collision rates
    if pairs in ["ice_ice","snow_snow"]:
        plot_coll_rates_selfcollection(axes,df,pairs)
    elif pairs in ["ice_snow"]:
        plot_coll_rates_icesnowcollection(axes,df,pairs)
