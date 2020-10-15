
'''
this "function" reads pamtra output ncdf-file
'''
#load modules necessary for reading data
import numpy as np
from netCDF4 import Dataset
from scipy.special import gamma
from __access_P3_lookuptable import read_and_access_p3lookup_table
import sys
import time
def allocate_hyd(iconData,num_lev):
	#create list with the names of num_lev hydrometeor categories
	numbers	=	map(str, range(num_lev))

	for lev in numbers:
		iconData["qc_" + lev]	=	np.zeros([1,num_lev])
		iconData["qr_" + lev]	=	np.zeros([1,num_lev])
		iconData["qi_1_" + lev]	=	np.zeros([1,num_lev])#create 4 hydrometeor cateogories for each box, due to non-uniform m-D relationship
		iconData["qi_2_" + lev]	=	np.zeros([1,num_lev])
		iconData["qi_3_" + lev]	=	np.zeros([1,num_lev])
		iconData["qi_4_" + lev]	=	np.zeros([1,num_lev])

	for lev in numbers:
		iconData["qnc_" + lev]	=	np.zeros([1,num_lev])
		iconData["qnr_" + lev]	=	np.zeros([1,num_lev])
		iconData["qni_1_" + lev]=	np.zeros([1,num_lev])
		iconData["qni_2_" + lev]=	np.zeros([1,num_lev])
		iconData["qni_3_" + lev]=	np.zeros([1,num_lev])
		iconData["qni_4_" + lev]=	np.zeros([1,num_lev])

	return (iconData)

def write_massmix_num_conc_to_pseudo_categories(iconData,num_lev):
	
	for lev in range(0,num_lev):
		for q in ['qc','qnc','qr','qnr','qi_1','qi_2','qi_3','qi_4','qni_1','qni_2','qni_3','qni_4']:
			iconData[q + '_' + str(lev)][0,lev]=iconData[q][0,lev]
	return (iconData)

def write_massmix_num_conc_to_pamtra_4D_var(iconData,num_lev):
	#write hydrometeors into one variable named hydro_cmpl for mass mixing ratios and hydro_num_cmpl for number
	hydrometcount	=	0 #begin with first hydrometeor for mass mixing ratio
	hydro_cmpl 	= 	np.zeros([1,1,num_lev,6*num_lev])#allocate
	numbers	=	map(str, range(num_lev))

	#this can be shortened		
	for lev in numbers:
		hydro_cmpl[:,:,:,hydrometcount]		=	iconData["qc_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_cmpl[:,:,:,hydrometcount]		=	iconData["qr_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_cmpl[:,:,:,hydrometcount]		=	iconData["qi_1_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_cmpl[:,:,:,hydrometcount]		=	iconData["qi_2_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_cmpl[:,:,:,hydrometcount]		=	iconData["qi_3_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_cmpl[:,:,:,hydrometcount]		=	iconData["qi_4_" + lev][0,:]
		hydrometcount				+=1
	#number concentrations
	hydrometcount	=	0 #begin with first hydrometeor for number concentration
	hydro_num_cmpl 	= 	np.zeros([1,1,num_lev,6*num_lev])#allocate
	for lev in numbers:
		hydro_num_cmpl[:,:,:,hydrometcount]		=	iconData["qnc_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_num_cmpl[:,:,:,hydrometcount]		=	iconData["qnr_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_num_cmpl[:,:,:,hydrometcount]		=	iconData["qni_1_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_num_cmpl[:,:,:,hydrometcount]		=	iconData["qni_2_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_num_cmpl[:,:,:,hydrometcount]		=	iconData["qni_3_" + lev][0,:]
		hydrometcount				+=1
	for lev in numbers:
		hydro_num_cmpl[:,:,:,hydrometcount]		=	iconData["qni_4_" + lev][0,:]
		hydrometcount				+=1
	
	return (hydro_cmpl,hydro_num_cmpl)


def calc_threshold(i_rhor,i_Fr,n_tot,qi,Fr,rho_rim):#taken from ICON & modified; D4 is an array with diameters for with aas4 should be calculated, if D4 is not known specify numbin and D4 will be created as a vector from dcritr to 0.01 with numbin_4 steps
	'''
	Davide's version that cuts some annoying parts to vectorize
	calculate threshold between different m-D regions and paramters a, b of the m-D-relationship if not constant
	'''
	###########################################################################################################################
	# Davide's variation to always have a return value
	if n_tot<=1 or qi<=1e-10:
		mui = np.nan
		lami = dcrit = dcrits = dcritr = cs1 = ds1 = cs = ds = cgp_now = dg = csr = dsr = aas1 = bas1 = aas2 = bas2 = aas3 = bas3 = 0.0
		#aas4 = np.zeros(numbin_4)
		#bas4 = np.nan
		return n_tot, mui, lami, dcrit, dcrits, dcritr, cs1, ds1, cs, ds, cgp_now, dg, csr, dsr, aas1, bas1, aas2, bas2, aas3, bas3#, aas4, bas4
	###########################################################################################################################
	time0 = time.time()
	#get lami from lookup-table #try if this works -> if yes lami can be replaced as an input (and dont have to be an output variable in the ICON-run)
	iData = dict();
	iData['rime_dens'] = rho_rim;iData['qi'] = qi;iData['qni'] = n_tot;iData['qirim'] = Fr*qi;iData['birim']= 1.0/rho_rim*iData['qirim']
	#iconData['qi']=q
	#iconData['qni']=N_tot
	#iconData['qinorm']=qinorm #copy to iconData to handle it within __access_P3_lookuptable as it would come from icon itself
	#set prognostic variables describing riming to zero or specify it here
	#iconData['qirim']= Frim*iconData['qi'] #np.zeros(N_tot.shape)
	#iconData['birim']= 1.0/rho_rim*iconData['qirim']*np.ones(N_tot.shape)#np.zeros(N_tot.shape) #set prognostic variables describing riming to zero
	#iconData['rime_dens'] = rho_rim
	#print 'time in calc_threshold_and_params_in_size_dist (0.5)',time.time()-time0
	iData = read_and_access_p3lookup_table(iData,['lami'])
	lami = iData['lami']
	#print 'time in calc_threshold_and_params_in_size_dist (1)',time.time()-time0
	#some constants
	pi  	=	3.14159265359
	sxth	=	1./6.
	#calc mu from lami division by 100 is to convert m-1 to cm-1
	mui = 0.076*(lami/100.)**0.8-2. #for ice mu is calculated inside pamtra
	# make sure mu_i >= 0, otherwise size dist is infinity at D = 0
	mui = max(mui,0.)
	# set upper limit at 6
	mui = min(mui,6.)
	##############################################
	####1:spherical ice m(D)=cs1*D**ds1###########
	##############################################
	#upper limit: dcrit=6.71e-05=67mu m (constant)
	##############################################

	#set up m-D relationship for solid ice with D < Dcrit
	cs1  = pi*sxth*900.
	ds1  = 3.
	#########################################
	####2:dense nonspherical m(D)=cs*D**ds###
	#########################################	
	#lower limit: dcrit, upper limit:dcrits##
	#########################################
	# Brown and Francis (1995)
	ds	=	1.9
	# cs = 0.01855 # original (pre v2.3), based on assumption of Dmax
	cs	=	0.0121 # scaled value based on assumtion of Dmean from Hogan et al. 2012, JAMC
	#########################################
	####3:graupel m(D)=cgp[i_rhor]*D**dg####
	#########################################
	#lower limit: dcrits, upper limit:dcritr#
	#########################################

	#specify m-D parameter for fully rimed ice
	#  note:  cg is not constant, due to variable density
	dg = ds1
	# alpha parameter of m-D for rimed ice
	crp	=	rho_rim*pi*sxth
	'''	
	crp	=	np.zeros(5)
	crp[0]	=	50.*pi*sxth
	crp[1]	=	250.*pi*sxth
	crp[2]	=	450.*pi*sxth
	crp[3]	=	650.*pi*sxth
	crp[4]	=	900.*pi*sxth
	'''
	#calculate first threshold between ...
	dcrit = (pi/(6.*cs)*900.)**(1./(ds-3.))
	# first guess, set cgp=crp
	#cgp		=	np.zeros(4)
       	#cgp[i_rhor] 	=	crp[i_rhor]
	cgp 		=	crp #[i_rhor]
	#print 'time in calc_threshold_and_params_in_size_dist (2)',time.time()-time0
	################################################
	####4:partially-rimed m(D)=csr[i_rhor]*D**dsr###
	################################################
	#lower limit: dcritr############################
	################################################
	#print 'i_Fr',i_Fr
	if (i_Fr==0 and Fr<1e-10):#consider index shift to Fortran code
		dcrits = 1.e+6
		dcritr = dcrits
		csr    = cs
		dsr    = ds
	# case of partial riming (Fr between 0 and 1)
	elif (i_Fr==0 or i_Fr==1 or i_Fr==2): #we have to make a difference to create_lookup_table "elif (i_Fr==1 or i_Fr==2) because we are not interpolating between this values in the end
		while True:#needs up to ~10 iterations -> could be speed up by a lookup-table
			#dcrits = (cs/cgp[i_rhor])**(1./(dg-ds))
			dcrits = (cs/cgp)**(1./(dg-ds))
			#dcritr = ((1.+Fr/(1.-Fr))*cs/cgp[i_rhor])**(1./(dg-ds))
			dcritr = ((1.+Fr/(1.-Fr))*cs/cgp)**(1./(dg-ds))
			csr    = cs*(1.+Fr/(1.-Fr))
			dsr    = ds

			#get mean density of vapor deposition/aggregation grown ice
			rhodep = 1./(dcritr-dcrits)*6.*cs/(pi*(ds-2.))*(dcritr**(ds-2.)-dcrits**(ds-2.))

			#get density of fully-rimed ice as rime mass fraction weighted rime density plus
			#density of vapor deposition/aggregation grown ice
			cgpold      = cgp #[i_rhor]
			#cgp[i_rhor] = crp[i_rhor]*Fr+rhodep*(1.-Fr)*pi*sxth
			cgp = crp*Fr+rhodep*(1.-Fr)*pi*sxth
	     		#if (abs((cgp[i_rhor]-cgpold)/cgp[i_rhor])<0.01):
			if (abs((cgp-cgpold)/cgp)<0.01):
				break
          
	# case of complete riming (Fr=1.0)
	else:
		#set threshold size between partially-rimed and fully-rimed ice as arbitrary large
		#dcrits = (cs/cgp[i_rhor])**(1./(dg-ds))
		dcrits = (cs/cgp)**(1./(dg-ds))
		dcritr = 1.e+6       # here is the "arbitrary large"
		csr    = cgp #[i_rhor]
		dsr    = dg
	#save current graupel coefficient
	cgp_now	=	cgp #[i_rhor]
	#print results
	#print '      dcrit,          dcrits,	     dcritr,       csr,     dsr,       mui,     lami,       cgp'
	#print dcrit,dcrits,	dcritr, csr  , dsr, mui, lami, cgp_now
	#print 'time in calc_threshold_and_params_in_size_dist (3)',time.time()-time0
	################
	#A-D-parameters
	################
	'''
	These are constants... why do I have to calculate them everytime?
	'''
	####1:spherical ice: 	 A(D)=aas1*D**bas1
	aas1=np.pi/4.
	bas1=2

	###2:dense nonspherical: A(D)=aas2*D**bas2
	bas2=1.88
	aas2=0.2285*100.**bas2/(100.**2)

	####3:graupel: 	     	 A(D)=aas3*D**bas3
	aas3=np.pi/4.
	bas3=2

	'''
	The following part returns a vector and causes problems to vectorization, but it is simple enough that can be done externally
	'''
	####4:partially-rimed:	 A(D)=aas4*D**bas4
#	bas4=1.88

	#calculate aas4 for the whole range of diameter where partially rimed crystals are present
#	if not isinstance(D4, np.ndarray): #any(D4==-99):
#		D4=np.logspace(np.log10(dcritr),-2.0,num=numbin_4,base=10)
#
#	aas4= 	np.zeros(D4.shape[0])
#	bincounter=0
#	for D_now in D4:
#		#calculate first the extrapolated areas with the A-D relationships of dense nonspherical(II) and graupel(III) at this diameter
#		A2=aas2*D_now**bas2#dense nonspherical
#		A3=aas3*D_now**bas3#graupel
#		A4=(1.-Fr)*A2+Fr*A3 #area of partially rimed crystals as an linear weighting by rimed fraction
#		#finally calculate aas4 for a specific diameter
#		aas4[bincounter]=A4/D_now**bas4
#		bincounter=bincounter+1
#
	return n_tot,mui,lami,dcrit,dcrits,dcritr,cs1,ds1,cs,ds,cgp_now,dg,csr,dsr,aas1,bas1,aas2,bas2,aas3,bas3#,aas4,bas4


def calc_threshold_and_params_in_size_dist(i_rhor,i_Fr,n_tot,qi,Fr,rho_rim,D4=-99,numbin_4=-99):#taken from ICON & modified; D4 is an array with diameters for with aas4 should be calculated, if D4 is not known specify numbin and D4 will be created as a vector from dcritr to 0.01 with numbin_4 steps
	'''
	calculate threshold between different m-D regions and paramters a, b of the m-D-relationship if not constant
	'''
	time0 = time.time()
	#get lami from lookup-table #try if this works -> if yes lami can be replaced as an input (and dont have to be an output variable in the ICON-run)
	iData = dict();
	iData['rime_dens'] = rho_rim;iData['qi'] = qi;iData['qni'] = n_tot;iData['qirim'] = Fr*qi;iData['birim']= 1.0/rho_rim*iData['qirim']
	#iconData['qi']=q
	#iconData['qni']=N_tot
	#iconData['qinorm']=qinorm #copy to iconData to handle it within __access_P3_lookuptable as it would come from icon itself
	#set prognostic variables describing riming to zero or specify it here
	#iconData['qirim']= Frim*iconData['qi'] #np.zeros(N_tot.shape)
	#iconData['birim']= 1.0/rho_rim*iconData['qirim']*np.ones(N_tot.shape)#np.zeros(N_tot.shape) #set prognostic variables describing riming to zero
	#iconData['rime_dens'] = rho_rim
	#print 'time in calc_threshold_and_params_in_size_dist (0.5)',time.time()-time0
	iData = read_and_access_p3lookup_table(iData,['lami'])
	lami = iData['lami']
	#print 'time in calc_threshold_and_params_in_size_dist (1)',time.time()-time0
	#some constants
	pi  	=	3.14159265359
	sxth	=	1./6.
	#calc mu from lami division by 100 is to convert m-1 to cm-1
	mui = 0.076*(lami/100.)**0.8-2. #for ice mu is calculated inside pamtra
	# make sure mu_i >= 0, otherwise size dist is infinity at D = 0
	mui = max(mui,0.)
	# set upper limit at 6
	mui = min(mui,6.)
	##############################################
	####1:spherical ice m(D)=cs1*D**ds1###########
	##############################################
	#upper limit: dcrit=6.71e-05=67mu m (constant)
	##############################################

	#set up m-D relationship for solid ice with D < Dcrit
	cs1  = pi*sxth*900.
	ds1  = 3.
	#########################################
	####2:dense nonspherical m(D)=cs*D**ds###
	#########################################	
	#lower limit: dcrit, upper limit:dcrits##
	#########################################
	# Brown and Francis (1995)
	ds	=	1.9
	# cs = 0.01855 # original (pre v2.3), based on assumption of Dmax
	cs	=	0.0121 # scaled value based on assumtion of Dmean from Hogan et al. 2012, JAMC
	#########################################
	####3:graupel m(D)=cgp[i_rhor]*D**dg####
	#########################################
	#lower limit: dcrits, upper limit:dcritr#
	#########################################

	#specify m-D parameter for fully rimed ice
	#  note:  cg is not constant, due to variable density
	dg = ds1
	# alpha parameter of m-D for rimed ice
	crp	=	rho_rim*pi*sxth
	'''	
	crp	=	np.zeros(5)
	crp[0]	=	50.*pi*sxth
	crp[1]	=	250.*pi*sxth
	crp[2]	=	450.*pi*sxth
	crp[3]	=	650.*pi*sxth
	crp[4]	=	900.*pi*sxth
	'''
	#calculate first threshold between ...
	dcrit = (pi/(6.*cs)*900.)**(1./(ds-3.))
	# first guess, set cgp=crp
	#cgp		=	np.zeros(4)
       	#cgp[i_rhor] 	=	crp[i_rhor]
	cgp 		=	crp #[i_rhor]
	#print 'time in calc_threshold_and_params_in_size_dist (2)',time.time()-time0
	################################################
	####4:partially-rimed m(D)=csr[i_rhor]*D**dsr###
	################################################
	#lower limit: dcritr############################
	################################################
	#print 'i_Fr',i_Fr
	if (i_Fr==0 and Fr<1e-10):#consider index shift to Fortran code
		dcrits = 1.e+6
		dcritr = dcrits
		csr    = cs
		dsr    = ds
	# case of partial riming (Fr between 0 and 1)
	elif (i_Fr==0 or i_Fr==1 or i_Fr==2): #we have to make a difference to create_lookup_table "elif (i_Fr==1 or i_Fr==2) because we are not interpolating between this values in the end
		while True:#needs up to ~10 iterations -> could be speed up by a lookup-table
			#dcrits = (cs/cgp[i_rhor])**(1./(dg-ds))
			dcrits = (cs/cgp)**(1./(dg-ds))
			#dcritr = ((1.+Fr/(1.-Fr))*cs/cgp[i_rhor])**(1./(dg-ds))
			dcritr = ((1.+Fr/(1.-Fr))*cs/cgp)**(1./(dg-ds))
			csr    = cs*(1.+Fr/(1.-Fr))
			dsr    = ds

			#get mean density of vapor deposition/aggregation grown ice
			rhodep = 1./(dcritr-dcrits)*6.*cs/(pi*(ds-2.))*(dcritr**(ds-2.)-dcrits**(ds-2.))

			#get density of fully-rimed ice as rime mass fraction weighted rime density plus
			#density of vapor deposition/aggregation grown ice
			cgpold      = cgp #[i_rhor]
			#cgp[i_rhor] = crp[i_rhor]*Fr+rhodep*(1.-Fr)*pi*sxth
			cgp = crp*Fr+rhodep*(1.-Fr)*pi*sxth
	     		#if (abs((cgp[i_rhor]-cgpold)/cgp[i_rhor])<0.01):
			if (abs((cgp-cgpold)/cgp)<0.01):
				break
          
	# case of complete riming (Fr=1.0)
	else:
		#set threshold size between partially-rimed and fully-rimed ice as arbitrary large
		#dcrits = (cs/cgp[i_rhor])**(1./(dg-ds))
		dcrits = (cs/cgp)**(1./(dg-ds))
		dcritr = 1.e+6       # here is the "arbitrary large"
		csr    = cgp #[i_rhor]
		dsr    = dg
	#save current graupel coefficient
	cgp_now	=	cgp #[i_rhor]
	#print results
	#print '      dcrit,          dcrits,	     dcritr,       csr,     dsr,       mui,     lami,       cgp'
	#print dcrit,dcrits,	dcritr, csr  , dsr, mui, lami, cgp_now
	#print 'time in calc_threshold_and_params_in_size_dist (3)',time.time()-time0
	################
	#A-D-parameters
	################
	####1:spherical ice: 	 A(D)=aas1*D**bas1
	aas1=np.pi/4.
	bas1=2

	###2:dense nonspherical: A(D)=aas2*D**bas2
	bas2=1.88
	aas2=0.2285*100.**bas2/(100.**2)

	####3:graupel: 	     	 A(D)=aas3*D**bas3
	aas3=np.pi/4.
	bas3=2

	####4:partially-rimed:	 A(D)=aas4*D**bas4
	bas4=1.88
	#calculate aas4 for the whole range of diameter where partially rimed crystals are present
	if not isinstance(D4, np.ndarray): #any(D4==-99):
		D4=np.logspace(np.log10(dcritr),-2.0,num=numbin_4,base=10)

	aas4= 	np.zeros(D4.shape[0])
	bincounter=0
	for D_now in D4:
		#calculate first the extrapolated areas with the A-D relationships of dense nonspherical(II) and graupel(III) at this diameter
		A2=aas2*D_now**bas2#dense nonspherical
		A3=aas3*D_now**bas3#graupel
		A4=(1.-Fr)*A2+Fr*A3 #area of partially rimed crystals as an linear weighting by rimed fraction
		#finally calculate aas4 for a specific diameter
		aas4[bincounter]=A4/D_now**bas4
		bincounter=bincounter+1

	return n_tot,mui,lami,dcrit,dcrits,dcritr,cs1,ds1,cs,ds,cgp_now,dg,csr,dsr,aas1,bas1,aas2,bas2,aas3,bas3,aas4,bas4


def get_rime_fraction_index(Frim):#taken from ICON & modified
	#size of rime fraction in lookupTable
	rimsize	=	4-1 #-1 due to index shift between fortran and python
	#find index for rime mass fraction
	#eps = 1e-10 #arbitrarily chosen
	#if (Frim < eps): #no rime present
	#	Frim_ind_float = 0.0
	#else:
	Frim_ind_float	= Frim*3.0 #index for rime fraction in lookupTable
	Frim_ind_int	= int(Frim_ind_float)#convert index to integer; both fortran and python round down to next lowest integer
	#set limits
	Frim_ind_float  = min(Frim_ind_float,rimsize) 	#index should be <=3
	Frim_ind_float  = max(Frim_ind_float,0.)      	#index should be >=0
	Frim_ind_int	= max(0,Frim_ind_int)		#index should be >=0
	Frim_ind_int 	= min(rimsize,Frim_ind_int)	#index should be <=3
	#print 'Frim',Frim
	#print 'Frim_ind_int',Frim_ind_int
	
	return Frim_ind_int


def get_rime_density_index(rho_rim):#taken from ICON & modified
	#size of rime density in lookupTable
	densize	=	5-1 #-1 due to index shift between fortran and python
	#limit rho_rim to [50;900]kg m-3 #thats done in calc_bulk_rime in mo_p3_mcrph.f90 for icon
	rho_rim	= min(rho_rim,899) #900 will result in an error because the index i_rhor is then 5
	rho_rim	= max(rho_rim,50)
	#find index for bulk rime density
        #(account for uneven spacing in lookup table for density)
	if (rho_rim<650.):
		rho_rim_ind_float = (rho_rim-50.)*0.005
	else:
		rho_rim_ind_float =(rho_rim-650.)*0.004 + 3.

	rho_rim_ind_int = rho_rim_ind_float
        # set limits
	rho_rim_ind_float  = min(rho_rim_ind_float,densize)
	rho_rim_ind_float  = max(rho_rim_ind_float,0.)
	rho_rim_ind_int = max(0,rho_rim_ind_int)
	rho_rim_ind_int = min(densize,rho_rim_ind_int)
	#print 'rho_rim'
	#print rho_rim
	#print 'rho_rim_ind_int'
	#print rho_rim_ind_int
	return rho_rim,int(rho_rim_ind_int)

def  calc_bulkRhoRime(qi_rim,bi_rim):#taken from ICON & modified

	##--------------------------------------------------------------------------------
	##  Calculates and returns the bulk rime density from the prognostic ice variables
	##  ((and adjusts qirim and birim appropriately.))
	##--------------------------------------------------------------------------------


	if (bi_rim>1.e-15):
		rho_rime = qi_rim/bi_rim
	    #impose limits on rho_rime;  adjust bi_rim if needed
		if (rho_rime<50):
			rho_rime = 50
			bi_rim   = qi_rim/rho_rime
		elif (rho_rime>900):
			rho_rime = 900
			bi_rim   = qi_rim/rho_rime
	else:
		qi_rim   = 0.
		bi_rim   = 0.
		rho_rime = 0.
		
	return rho_rime


def pseudo_adaptive_grid(N_tot,lam,mu):#try to calculate an own adaptive grid, because we cannot use the pamtra adaptive grid due to the four categories of ice
	d_low	=	np.zeros(N_tot.shape[0])
	d_high	=	np.zeros(N_tot.shape[0])
	diam	=	np.logspace(-12.0,-1.0,num=200,base=10)	#list of diameter;
	thresh	=	0#10**-100#threshold of concentration which should be exceeded
	for i in range(1,N_tot.shape[0]):
		lam_now	=	lam[i]
		N_0_now	=	N_tot[i] / gamma(mu[i] + 1) * lam_now**(mu[i] + 1)
		mu_now	=	mu[i]
		N_D	=	N_0_now*diam**mu_now*np.exp(-lam_now*diam)
		if any(N_D>thresh):
			first	=	next(x[0] for x in enumerate(N_D) if x[1] > thresh) #get first element which exceed limit
			last	=	N_tot.shape[0]-next(x[0] for x in enumerate(reversed(N_D)) if x[1] > thresh)-1 #get first element which exceed limit		
		else:
			first	=	0
			last	=	0

		d_low[i]	=	diam[first]
		d_high[i]	=	diam[last]
		#print 'd_low: '  + str(d_low[i])
		#print 'd_high: ' + str(d_high[i])		
	return d_low,d_high

def  get_cloud_dsd(qc,nc,pres,temp):

	# parameters for droplet mass spectral shape, used by Seifert and Beheng (2001)
	# warm rain scheme only (iparam = 1)
	dnu = np.zeros([17])
	dnu[1]  =  0.
	dnu[2]  = -0.557
	dnu[3]  = -0.430
	dnu[4]  = -0.307
	dnu[5]  = -0.186
	dnu[6]  = -0.067
	dnu[7]  =  0.050
	dnu[8]  =  0.167
	dnu[9]  =  0.282
	dnu[10] =  0.397
	dnu[11] =  0.512
	dnu[12] =  0.626
	dnu[13] =  0.739
	dnu[14] =  0.853
	dnu[15] =  0.966
	dnu[16] =  0.966

	qsmall = 1.e-14
	nsmall = 1.e-16
	rd     = 287.15
	rho = pres/(rd*temp)
	rhow = 997.
	cons1=np.pi/6*rhow
	thrd = 1./3.
	if qc>qsmall:

		# set minimum nc to prevent floating point error
		nc   = np.maximum(nc,nsmall)
		mu_c = 0.0005714*(nc*1.e-6*rho)+0.2714
		mu_c = 1./(mu_c**2)-1.
		mu_c = np.maximum(mu_c,2.)
		mu_c = np.minimum(mu_c,15.)

		# interpolate for mass distribution spectral shape parameter (for SB warm processes)
		iparam = 3
		if iparam == 1:
			dumi = int(mu_c)
			nu   = dnu[dumi]+(dnu[dumi+1]-dnu[dumi])*(mu_c-dumi)

		# calculate lamc
		lamc = (cons1*nc*(mu_c+3.)*(mu_c+2.)*(mu_c+1.)/qc)**thrd

		# apply lambda limiters
		lammin = (mu_c+1.)*2.5e+4   # min: 40 micron mean diameter
		lammax = (mu_c+1.)*1.e+6    # max:  1 micron mean diameter

		if lamc < lammin:
			lamc = lammin
			nc   = 6.*lamc**3*qc/(np.pi*rhow*(mu_c+3.)*(mu_c+2.)*(mu_c+1.))
		elif lamc>lammax:
			lamc = lammax
			nc   = 6.*lamc**3*qc/(np.pi*rhow*(mu_c+3.)*(mu_c+2.)*(mu_c+1.))

		cdist  = nc*(mu_c+1.)/lamc
		cdist1 = nc/gamma(mu_c+1.)

	else:
		mu_c   = 0.
		lamc   = 0.
		cdist  = 0.
		cdist1 = 0.
	
	#end subroutine get_cloud_dsd
	return mu_c,lamc

def get_rain_dsd(qr,nr,pres,temp,mu_r_table): #,mu_r,rdumii,dumii,lamr,mu_r_table,cdistr,logn0r,log_qrpresent,qrindex,k)
	# Computes and returns rain size distribution parameters
	qsmall = 1.e-14
	nsmall = 1.e-16
	rd     = 287.15
	rho = pres/(rd*temp)
	rhow = 997.
	cons1=np.pi/6*rhow
	thrd = 1./3.

	if qr>qsmall:

		# use lookup table to get mu
		# mu-lambda relationship is from Cao et al. (2008), eq. (7)

		# find spot in lookup table
		# (scaled N/q for lookup table parameter space_
		nr      = max(nr,nsmall)
		inv_dum = (qr/(cons1*nr*6.))**thrd

		if (inv_dum<282e-6):
		     	mu_r = 8.282
		elif (inv_dum>282.e-6 and inv_dum<502.e-6):
		   	# interpolate
			rdumii   = (inv_dum-250.e-6)*1.e+6*0.5
			rdumii   = max(rdumii,1.)
			rdumii   = min(rdumii,150.)
			dumii    = int(rdumii)
			dumii    = min(149,dumii)
			mu_r     = mu_r_table[dumii]+(mu_r_table[dumii+1]-mu_r_table[dumii])*(rdumii-float(dumii))
		elif (inv_dum>502.e-6):
			mu_r = 0.
		#endif

		lamr = (cons1*nr*(mu_r+3.)*(mu_r+2)*(mu_r+1.)/(qr))**thrd  # recalculate slope based on mu_r
		lammax = (mu_r+1.)*1.e+5   # check for slope
		lammin = (mu_r+1.)*1250.   # set to small value since breakup is explicitly included (mean size 0.8 mm)

		# apply lambda limiters for rain
		if (lamr<lammin):
			lamr = lammin
			nr   = np.exp(3.*np.log(lamr)+np.log(qr)+np.log(gamma(mu_r+1.))-np.log(gamma(mu_r+4.)))/(cons1)
		elif (lamr>lammax):
			lamr = lammax
			nr   = np.exp(3.*np.log(lamr)+np.log(qr)+np.log(gamma(mu_r+1.))-np.log(gamma(mu_r+4.)))/(cons1)
		#endif

		 
		cdistr = nr/gamma(mu_r+1.)
		logn0r    = np.log10(nr)+(mu_r+1.)*np.log10(lamr)-np.log10(gamma(mu_r+1)) #note: logn0r is calculated as log10(n0r)

	else:
		mu_r = 0.
		lamr = 0.
		cdistr = 0.
		logn0r = 0.

	#endif

	#end subroutine get_rain_dsd
	return mu_r,lamr

def generate_rain_lookuptable():
#------------------------------------------------------------------------------------------#

	thrd = 1./3.
	# Generate lookup table for rain shape parameter mu_r
	# this is very fast so it can be generated at the start of each run
	# make a 150x1 1D lookup table, this is done in parameter
	# space of a scaled mean size proportional qr/Nr -- initlamr

	#print*, '   Generating rain lookup-table ...'
	mu_r_table = np.zeros(151)
	for i in  range(1,150+1):              # loop over lookup table values
		initlamr = 1./((float(i)*2.)*1.e-6 + 250.e-6)

		# iterate to get mu_r
		# mu_r-lambda relationship is from Cao et al. (2008), eq. (7)

		# start with first guess, mu_r = 0

		mu_r = 0.
		#converged = False
		for ii in range(1,50+1):
			lamr = initlamr*((mu_r+3.)*(mu_r+2.)*(mu_r+1.)/6.)**thrd

			# new estimate for mu_r based on lambda
			# set max lambda in formula for mu_r to 20 mm-1, so Cao et al.
			# formula is not extrapolated beyond Cao et al. data range
			dum  = min(20.,lamr*1.e-3)
			mu_r = max(0.,-0.0201*dum**2+0.902*dum-1.718)

			# if lambda is converged within 0.1%, then exit loop
			if (ii>2):
				if (abs((lamold-lamr)/lamr)<0.001): break
			lamold = lamr


		#111 continue

		# assign lookup table values
		mu_r_table[i] = mu_r
	np.savez('mu_r_table.npz',mu_r_table=mu_r_table)
	return mu_r_table
		#enddo


def rh2vap(temp_p,pres_p,rh):
	#reverse vap2rh from pamtras conversions.f90

	tpt = 273.16               # triple point temperature
	estpt = 611.14             # saturation vapor triple point temperature
	r_d = 287.0596736665907    #gas constant of dry air
	r_v = 461.5249933083879    #gas constant of water vapor
	mmv = 18.0153e-3           # molar mass of vapor
	mmd = 28.9644e-3           # molar mass of dry air
	vapor_hc  = 2.5008e+6      # vaporization heat constant
	sublim_hc  = 2.8345e+6     # sublimation heat constant

	'''
        !     XPABSM air pressure in Pa
        !     ZTEMP air temperature in K
        !     XRM water vapor mass mixing ratio kg/kg


        REAL(kind=dbl) :: XCPV               ! Cpv (vapor)
        REAL(kind=dbl) :: XCL,XCI            ! Cl (liquid), Ci (ice)
        REAL(kind=dbl) :: XLMTT              ! Melting heat constant
        REAL(kind=dbl) :: XALPW,XBETAW,XGAMW ! Const saturation vapor pressure  (liquid)
        REAL(kind=dbl) :: XALPI,XBETAI,XGAMI ! Consts saturation vapor pressure  (solid ice)
        real(kind=dbl) :: ztemp,xrm,xpabsm,zwork31,zwork32
	'''

	zwork32 = 0
	XCPV   = 4. * r_v
	XCL    = 4.218e+3
	XCI    = 2.106e+3
	XLMTT  = sublim_hc - vapor_hc
	XGAMW  = (XCL - XCPV) / r_v
	XBETAW = (vapor_hc/r_v) + (XGAMW * tpt)
	XALPW  = np.log(sublim_hc) + (XBETAW /tpt) + (XGAMW *np.log(tpt))
	XGAMI  = (XCI - XCPV) / r_v
	XBETAI = (sublim_hc/r_v) + (XGAMI * tpt)
	XALPI  = np.log(estpt) + (XBETAI /tpt) + (XGAMI *np.log(tpt))

	ZTEMP = temp_p
	#XRM = hum_massmix
	XPABSM = pres_p

        #!     * humidit351 relative par rapport 340 l'eau liquide
	''' #original code: but we want to get qv from rh, so the other way round 
        if (ZTEMP >= tpt):
            ZWORK31=np.exp(XALPW - XBETAW/ZTEMP - XGAMW*np.log(ZTEMP))
            ZWORK31=(mmv/mmd)*ZWORK31/(XPABSM-ZWORK31)
            ZWORK32=100.*XRM/ZWORK31
        elif (ZTEMP < tpt):
            #!     * humidit351 relative par rapport 340 la glace
            ZWORK31=np.exp(XALPI - XBETAI/ZTEMP - XGAMI*np.log(ZTEMP))
            ZWORK31=(mmv/mmd)*ZWORK31/(XPABSM-ZWORK31)
            ZWORK32=100.*XRM/ZWORK31
		#if you skip the second line
		#ZWORK32=100.*XRM/((mmv/mmd)*ZWORK31/(XPABSM-ZWORK31)) (Eq 1)

        vapor2rh=ZWORK32
	'''
	'''
	if (ZTEMP >= tpt): #naming as here: https://github.com/PyAOS/aoslib/blob/master/aoslib/src/spechum.f
		#calculate vapor pressure
		ZWORK31 = np.exp(XALPW - XBETAW/ZTEMP - XGAMW*np.log(ZTEMP))
		#reverse Eq 1
		hum_massmix = 1./100.* rh * ((mmv/mmd)*ZWORK31/(XPABSM-ZWORK31))
        elif (ZTEMP < tpt):
		ZWORK31 = np.exp(XALPI- XBETAI/ZTEMP - XGAMI*np.log(ZTEMP))
		#reverse Eq 1
		hum_massmix = 1./100.* rh * ((mmv/mmd)*ZWORK31/(XPABSM-ZWORK31))
	'''
	ZWORK31 = np.where(ZTEMP >= tpt,np.exp(XALPW - XBETAW/ZTEMP - XGAMW*np.log(ZTEMP)),np.exp(XALPI- XBETAI/ZTEMP - XGAMI*np.log(ZTEMP))) #translated the if clause above into pythonic handling of arrays
	#reverse Eq 1
	hum_massmix = 1./100.* rh * ((mmv/mmd)*ZWORK31/(XPABSM-ZWORK31))

	return hum_massmix

def q2abs(spec_var,t,p,rh,q_all_hydro):#input specific variable to convert [kg/kg], temperatur [K], pressure [Pa], specific humidity [kg/kg], sum of specific masses of all hydrometeors [kg/kg]
	#taken from pamtras conversions.f90
	#use constants, only: r_d, r_v
	#real(kind=dbl), parameter :: r_d = 287.0596736665907_dbl    ! gas constant of dry air
	#real(kind=dbl), parameter :: r_v = 461.5249933083879_dbl    ! gas constant of water vapor
	r_d = 287.0596736665907    #gas constant of dry air
	r_v = 461.5249933083879    #gas constant of water vapor

	qv = rh2vap(t,p,rh) # get specific humidity [kg/kg]

	q2abs = spec_var*p/(r_d*(1.+(r_v/r_d-1.)*qv-q_all_hydro)*t)


	return q2abs
