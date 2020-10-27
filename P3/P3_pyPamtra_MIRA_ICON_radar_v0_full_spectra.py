# on fh2: devel/python/2.7_intel and lib/netcdf-fortran/4.4_serial should be loaded
# Python PAMTRA for ICON generator
import pyPamtra
import numpy as np
import math
from pprint import pprint			#print ncdf-file information (kind of header)
from netCDF4 import Dataset			#reading ncdf
from scipy.spatial.distance import cdist 	#calculate shortest distance
from scipy.special import gamma			#calculate gamma function
import sys
import os 					#import environment variables
import datetime #for conversion between unixtime and calender time
# Has to be modified, because actually python has to be started from meteoSI runtime directory
import imp
#meteoSI_func = imp.load_source('meteoSI_func', '/home/kit/imk-tro/ws3361/lib/python/pyPamtra/meteoSI.py')
#import self-written functions
from __prepare_hydrometeors import allocate_hyd,write_massmix_num_conc_to_pseudo_categories,write_massmix_num_conc_to_pamtra_4D_var,get_rime_fraction_index,get_rime_density_index,calc_threshold_and_params_in_size_dist,calc_bulkRhoRime,get_cloud_dsd,get_rain_dsd,generate_rain_lookuptable,q2abs
from __plot_size_dist_P3_ice_phase import plot_size_dist_P3_ice_phase
import time
#define some self-written functions here
def str2bool(v):#read booleans
  return v in ("True") #patterns listed here will return true

def read_subset_ncdf_var2D_forpamtra(Data,output_file,varlistin,varlistout,cell,num_files,filecount,filestart,timestep_per_file,timecount):
	#reads 'varlistin' variables from netcdf-file 'output_file' and writes it into 'varlistout'
	out 	= 	Dataset(output_file, mode='r') #open file
	if (filecount-filestart)==0:#initialize at first call
		#Data	=	dict()	#generate dictionary
		for varout in varlistin:
			Data[varout]	=	np.zeros([(timestep_per_file*(num_files))])#allocate matrices
	for varin,varout in zip(varlistin,varlistout):#read files and write it with different names in Data
		#Data[varout][None,:]	= np.squeeze(out.variables[varin][timecount,cell]) #rea
		if out.variables[varin].ndim == 2: #icon grid has 2 dimensions
			Data[varout][None,:]	= np.squeeze(out.variables[varin][timecount,cell])
		elif out.variables[varin].ndim == 3: #interpolated grid has one dimension more; be careful: y-dimension is just set to first index, because until now I just used pseudo-meteogram (4cell) lat-lon-grid
			Data[varout][None,:]	= np.squeeze(out.variables[varin][timecount,cell,0])	
	return Data

def read_subset_ncdf_var_forpamtra(Data,output_file,varlistin,varlistout,num_files,num_lev,filecount,filestart,timestep_per_file,timecount,lat_site,lon_site):
	#reads 'varlistin' variables from netcdf-file 'output_file' and writes it into 'varlistout'
	out 	= 	Dataset(output_file, mode='r') #open file
	if (filecount-filestart)==0:#initialize at first call
		for varout in varlistin:
			Data[varout]	=	np.zeros([(timestep_per_file*(num_files)),num_lev])#allocate matrices
	for varin,varout in zip(varlistin,varlistout):#read files and write it with different names in Data
		if out.variables[varin].ndim == 3: #icon grid has 3 dimensions
			Data[varout][None,:]	= np.squeeze(out.variables[varin][timecount,:,cell])
		elif out.variables[varin].ndim == 4: #interpolated grid has one dimension more; be careful: y-dimension is just set to first index, because until now I just used pseudo-meteogram (4cell) lat-lon-grid
			Data[varout][None,:]	= np.squeeze(out.variables[varin][timecount,:,lat_site,lon_site])
		#Data[varout][None,:]	= np.squeeze(out.variables[varin][timecount,:,cell])#read one column from current timestep	
	return Data
def create_unix_time(fh_ICON,writing_interval,timecount): #create unix-time from icon timestamp
	time_calender	= 	str(fh_ICON.variables["time"][timecount])
	#print 'timestamp icon: ' + time_calender#until now: read this time and put it in next line
	#get unix-time (seconds from new year 1970)
	#deal with the time code etc
	minutes = 0 #default if timestring has no character after the dot
	#print int(time_calender[0:4]),int(time_calender[4:6]),int(time_calender[6:8]),time_calender[9:10]
	for i in range(9,len(time_calender)):
		#print 'i=' + str(i) + 'timecal: ' + time_calender[i]
		minutes	=minutes+(10**(9-i+3))*float(time_calender[i])
	seconds			=	int((minutes/7*60)%60)
	minutes			=	int(minutes/7)%60
	hours			=	minutes//60
	#print 'seconds: ' + str(seconds)
	#print 'hours: ' + str(hours) +'minutes: ' + str(minutes)+'seconds: ' + str(seconds)
	#print 'minutes: ' + str(minutes)
	unixtime	=	(datetime.datetime(int(time_calender[0:4]),int(time_calender[4:6]),int(time_calender[6:8]),hours,minutes,seconds) - datetime.datetime(1970,1,1)).total_seconds()

	#print 'unixtime' + str(unixtime)
	return unixtime

#def distance(p1, p2):#simple pythagorian distance
#	return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def closest(pt, others):#find closest point in 'others' to pt
	#distances = distance(pt, others)
	distances = cdist(pt, others)#python build in for distances
	return distances.argmin() #returns cells with shortest distance

def find_cell_by_coordinates(lat,lon,lat_list,lon_list):
	lat_rad	=	np.deg2rad(lat)
	lon_rad	=	np.deg2rad(lon)
	cell_closest_to_site	=	closest(np.array([[lat_rad,lon_rad]]),zip(lat_list,lon_list))
	return cell_closest_to_site
#start timing at beginning of script
time0= time.time()
# Path to descriptor file
#desc_pamtra_file = 'descriptorfiles/descriptor_file_P3.txt'

#switches given by bash script
rad_freq		=	float(os.environ["rad_freq"]) #radar frequency
writing_interval	=	float(os.environ["writing_interval"]) #get output interval in minutes from bash script
onlycat			=	os.environ["onlycat"]#set all other than onlycat to 0
#onlyice			=	str2bool(os.environ["onlyice"])#set all other hydrometeors than qi to zero 
#onlyireg		=	int(os.environ["onlyireg"])#set all other than onlyireg to 0
#onlyliq			=	str2bool(os.environ["onlyliq"])#set all other hydrometeors than qc and qr to zero
#onlylcat		=	int(os.environ["onlylcat"])#set all other than onlylcat (rain or cloud droplet) to 0
debugging		=	str2bool(os.environ["debugging"])#higher verbosity etc
filestart		=	int(os.environ["filestart"])  #organize reading of icon files
fileend			=	int(os.environ["fileend"])
num_files		= fileend - filestart
timestep_per_file	=	int(os.environ["timesteps_per_file"])
#num_lev			=	int(os.environ["num_lev"])#number of vertical levels
icon_output_file_pre	=	os.environ["icon_output_file"]#name of icon output file without number at the end
cell_deg_bool		=	str2bool(os.environ["cell_deg_bool"])#True-> specify cell (position in icon output); False specify lat_site and lon_site
cell			=	int(os.environ["cell"])
lat_site		=	float(os.environ["lat_site"])
lon_site		=	float(os.environ["lon_site"])
#zres			=	int(os.environ["zres"]) #resolution of vertical axis in m; set to -99 if you do not use height as z-axis
# Initialize PyPAMTRA instance
pam = pyPamtra.pyPamtra()
nbins=100
##define hydrometeor categories: there can be just one scattering routine per category -> 2 categories for P3 ice to represent particle properties 
#!name       as_ratio    liq_ice     rho_ms    a_ms    b_ms    alpha    beta   moment_in   nbin      dist_name        p_1     p_2     p_3     p_4     d_1       d_2           scat_name   vel_size_mod           canting
pam.df.addHydrometeor(('cloudwater',  1.0,           1,        -99.,      -99.,    -99.,   -99.,     -99.,   13,           nbins,       'dummy',          -99.,    -99.,     -99.,   -99.,  -99.,    -99.        ,'mie-sphere',  'khvorostyanov01_drops',        -99))
pam.df.addHydrometeor(('rain',  1.0,           1,        -99.,      -99.,    -99.,   -99.,     -99.,   13,           nbins,       'dummy',          -99.,    -99.,     -99.,   -99.,  -99.,    -99.        ,'mie-sphere',  'khvorostyanov01_drops',        -99))
pam.df.addHydrometeor(('ice_sper',  1.0,           -1,        -99.,      -99.,    -99.,   -99.,     -99.,   13,           nbins,       'dummy',          -99.,    -99.,     -99.,   -99.,  -99.,    -99.        ,'mie-sphere',  'heymsfield10_particles',        -99))
pam.df.addHydrometeor(('ice_nonspher',  0.6,           -1,        -99.,      -99.,    -99.,   -99.,     -99.,   13,           nbins,       'dummy',          -99.,    -99.,     -99.,   -99.,  -99.,    -99.        ,'ss-rayleigh-gans',  'heymsfield10_particles',        0.))

#print 'time before loading mu_r_table',time.time()-time0
#get rain lookup-table
generate_rain_lookuptable() #generate rain lookup-table at beginning 
mu_r_table = np.load('mu_r_table.npz') #just read the lookup-table
mu_r_table=mu_r_table['mu_r_table']
#print 'time after loading mu_r_table',time.time()-time0
#turn off passive calculations
pam.nmlSet["passive"] = False  # Activate this for Microwave radiometer
pam.nmlSet["active"] = True    # Activate this for Cloud radar
pam.nmlSet["radar_mode"] = "moments"
pam.nmlSet["save_psd"] = False    # save particle size distribution
pam.nmlSet["radar_attenuation"] = "disabled" #"bottom-up"
pam.nmlSet["hydro_adaptive_grid"] = False    # uncomment this line before changing max and min diameter of size distribution
pam.nmlSet["conserve_mass_rescale_dsd"] = False    #necessary because of separating into m-D-relationship regions
pam.nmlSet["hydro_fullspec"] = True #use full-spectra as input
pam.nmlSet["radar_allow_negative_dD_dU"] = True #allow negative dU dD which can happen at the threshold between different particle species
pam.nmlSet["radar_pnoise0"]=-100 #set radar noise to an arbitrary low number
#pam.nmlSet["radar_nfft"] = 64 #256 #1024

#from Davide (but radar_fwhr_beamwidth_deg=0.5 for lacros MIRA-K)
pam.nmlSet["radar_fwhr_beamwidth_deg"] = 0.5 #check with ncdump -h /home/mkarrer/Dokumente/HOPE_data/20130526_krauthausen_categorize.nc
pam.nmlSet["radar_integration_time"] = 2.0
pam.nmlSet["radar_max_v"] = 10.56824
pam.nmlSet["radar_min_v"] = -10.56824
pam.nmlSet["radar_nfft"] = 512
pam.nmlSet["radar_pnoise0"] = -38.4
pam.nmlSet["radar_no_ave"] = 20
pam.nmlSet['radar_peak_min_snr'] = -20 


#show some messages
if debugging:
	print 'debugging'
	pam.set["verbose"] = 5
else:
	pam.set["verbose"] = 0
pam.set["pyVerbose"] = 0

########
###start timeloop
########
#if there are different files-> loop over files
filecounter	=	0 #for resuming after killed process: type here filestart(resuming)-filnumstart(not resumed) (this should be the first digit of the last example_ICON_radar_P3_... (f.e 543 -> 5) ; not resuming: set to 0
#print 'time before filecount-loop',time.time()-time0
iconData=dict()#needed for first call of 'read_subset_ncdf_var'
for filecount in range(filestart,fileend):
	icon_output_file  = icon_output_file_pre + str(filecount) + '.nc'
	print "reading file" + icon_output_file
	# Open NetCDF connection to file in read-only mode
	fh_ICON = Dataset(icon_output_file, mode='r')
	#pprint(fh_ICON) #print some information and varible list+dimension
	num_lev=fh_ICON.variables['temp'].shape[1] #; sys.exit
	#create list with the names of num_lev hydrometeor categories
	#(iconData) = allocate_hyd(iconData,num_lev)

	#read some time-invariant variables (grid)
	try:#latlon files (pseudo-meteogram) dont have  lat lon until now
		lon_icon	= 	fh_ICON.variables["clon"]
		lat_icon	= 	fh_ICON.variables["clat"]
	except:#take lat lon from HOPE
		lon_icon	=	50.909
		lat_icon	=	6.411
	if not cell_deg_bool: #search for nearest cell if cell is not given directly but lat and lon
		cell	=	find_cell_by_coordinates(lat_site,lon_site,lat_icon,lon_icon)
		print 'cell determined by lat lon: ' + str(cell)
	#zres=130
	#height		= 	np.arange(0,12000,zres) #here for output on z-levels old:#fh_ICON.variables["height"]

	height = fh_ICON.variables['height'][::-1] #read height directly
	var_list	= ["temp","pres","rh","qc","qi","qr","qnc","qni","qnr","qirim","birim"]
	timecounter	=	0
	#if debugging:
		#timestep_per_file=9#chose timestep to process here
		#starttime=timestep_per_file-1#process just one timestep
		#fileend=filestart #do not go process next file
	#else:
	starttime=0
	#print 'time before timecount-loop',time.time()-time0
	for timecount in range(starttime, timestep_per_file): #due to this loop all variables will appear as [0,:]
		print "timestep: " + str(timecount)
		###############################
		###allocate and read icon data
		###############################		
		unixtime	=	create_unix_time(fh_ICON,writing_interval,timecount)
		#read icon variables		
		(iconData)	=	read_subset_ncdf_var_forpamtra(iconData,icon_output_file,var_list,var_list,num_files,num_lev,filecount,filestart,timestep_per_file,timecount,lat_site,lon_site)
		#print 'time after read_subset_ncdf_var_forpamtra',time.time()-time0
		
		reverse_and_fill_vectors=True
		if reverse_and_fill_vectors:
			#pprint(iconData)
			#temp(always>0) is the first array and will be used here for masking
			#there are values for z<elevation which have approximately -9.98999994e-08 for all variables
			elements2fill = iconData['temp'][0,:]<0
			last_defined_ind = np.where(iconData['temp'][0,:]>0)[0]
			for key in iconData.keys():
				#print key,iconData[key].shape
				#fill element with next defined value
				last_defined = iconData[key][0,last_defined_ind][-1]
				#last_defined = iconData[key][0,np.where(iconData['temp'][0,:]>0)[0]][-1]	
				iconData[key][0,elements2fill] = last_defined
				#reverse vectors
				#if timecount==starttime: #flip only once
				if iconData[key].ndim>1:
					#flipped=np.fliplr(iconData[key])
					#print key,'flip'
					iconData[key][0,:] = iconData[key][0,::-1]#flipped
			#if timecount==starttime: #flip only once
			#	height=height[::-1]

		#print 'height',height[:]
		#print iconData['temp'][0,:]
		#read icon 2D variables	
		#var_list_2D	= ["t_g"]
		#(iconData)	=	read_subset_ncdf_var2D_forpamtra(iconData,icon_output_file,var_list_2D,var_list_2D,cell,num_files,filecount,filestart,timestep_per_file,timecount)
		#print 'time before  calculate shape parameters',time.time()-time0

		#calculate shape parameters
		#1. for cloud droplets
		iconData["mu_c"] = np.zeros(iconData['qc'].shape)
		iconData["lamc"] = np.zeros(iconData['qc'].shape)
		for i in range(0,iconData['qc'][0,:].size):
			if iconData['qc'][0,i]>1e-10 and iconData['qnc'][0,i]>1: 
				[iconData["mu_c"][0,i],iconData["lamc"][0,i]] = get_cloud_dsd(iconData['qc'][0,i],iconData['qnc'][0,i],iconData['pres'][0,i],iconData['temp'][0,i])
		#2. for rain drops
		iconData["mu_r"] = np.zeros(iconData['qr'].shape)
		iconData["lamr"] = np.zeros(iconData['qr'].shape)
		for i in range(0,iconData['qr'][0,:].size):
			if iconData['qr'][0,i]>1e-10 and iconData['qnr'][0,i]>1: 
				[iconData["mu_r"][0,i],iconData["lamr"][0,i]] = get_rain_dsd(iconData['qr'][0,i],iconData['qnr'][0,i],iconData['pres'][0,i],iconData['temp'][0,i],mu_r_table)
		#3. for the ice category this is done later because it requires additional variables like 'rime_dens' but the allocation is made here
		iconData["lami"] = np.zeros(iconData['qi'].shape)
		#print 'time before  calculate shape parameters',time.time()-time0
		#add qirim to qi to get full ice-phase mass
		iconData["qi"][0,:]=iconData["qi"][0,:]+iconData["qirim"][0,:]

		''' #not needed any more
		#link qni to qni_... for getting the same distribution parameters for all m-D regions
		iconData["qni_1"]=iconData["qni"]#be careful: this is a link-> change in qni_1 will change qni
		iconData["qni_2"]=iconData["qni"]
		iconData["qni_3"]=iconData["qni"]
		iconData["qni_4"]=iconData["qni"]
		'''
		#plot size distribution including threshold separating m-D relationship to know whats going on		
		debughere=False
		if debughere:
			for plotlev in range(1,num_lev):
				print 'qi level: ' + str(plotlev)
				print iconData["qi"][0,plotlev]
				if (iconData["qi"][0,plotlev]>10**-10) and plotlev==79: # and filecount==0 and timecount==0 and plotlev==50 and debughere:
					Frim  		= iconData["qirim"][0,plotlev]/iconData["qi"][0,plotlev] #rime fraction # qi here already is qi=qi+qirim!!
					i_Frim	=	get_rime_fraction_index(Frim)

					rho_rim	= (iconData["qirim"][0,plotlev]/iconData["birim"][0,plotlev]) #rime density
					[dum,i_rhor]	=	get_rime_density_index(rho_rim)
					print 'i_rhor' +str(i_rhor)		
					print 'rho_rim' +str(rho_rim)		
					print 'i_Frim' +str(i_Frim)		
					print 'Frim' +str(Frim)		
					plt	=	plot_size_dist_P3_ice_phase(i_rhor,rho_rim,i_Frim,iconData["lami"][0,plotlev],iconData["qni"][0,plotlev],Frim,iconData["qi"][0,plotlev])
					plt.savefig('output/plots/size_dist_t_' + str(filecount-filestart) + "%02d" % (timecount) + '/tmp_size_dist_lev' + str(plotlev) + '.pdf')
					plt.clf() #clear figure
					sys.exit() #uncomment to stop after generation of first plot
					continue
			continue
		#print 'time after calculate shape parameters',time.time()-time0
		###allocate thresholds and m-D-relationship parameter as vectors for all levels
		#allocate threshold vector for column
		#dcrit_col is a constant value
		dcrits_col	=	np.full(num_lev, np.nan) #separating nonspherical dense ice and graupel
		dcritr_col	=	np.full(num_lev, np.nan) #separating graupel and partially-rimed ice

		#allocate coefficents in m-D (a*D^b) relationship
		cgp_col	=	np.full(num_lev, np.nan) #a for graupel
		csr_col	=	np.full(num_lev, np.nan) #a for partially-rimed ice 

		#allocate coefficents in A-D (aas*D^bas) relationship
		aas4_col = 	np.full((num_lev,nbins), np.nan) #aas for partially-rimed ice
		#loop for calculating thresholds and m-D-relationship parameters
		for lev in range(0,num_lev):
			if (iconData["qi"][0,lev])>0: #avoid producing nans in case qi is zero
				Frim_now  	= iconData["qirim"][0,lev]/iconData["qi"][0,lev] #calculate rime fraction # qi here already is qi=qi+qirim!!
			else:
				Frim_now	=	0

			i_Frim_now		= get_rime_fraction_index(Frim_now)# there are 4 case for different sections of rime fraction

			rho_rim_now	=  calc_bulkRhoRime(iconData["qirim"][0,lev],iconData["birim"][0,lev]) #calculate rime density

			[rho_rim_now,i_rhor_now]	= get_rime_density_index(rho_rim_now)# there are 5 case for different sections of rime density
			#print 'time before  calc_threshold_and_params_in_size_dist',time.time()-time0
			#get parameters of m-D relationship-regions
			if iconData["qni"][0,lev]>1 and iconData["qi"][0,lev]>1e-10:
				[n_tot,mui,iconData["lami"][0,lev],dcrit,dcrits,dcritr,cs1,ds1,cs,ds,cgp_now,dg,csr,dsr,aas1,bas1,aas2,bas2,aas3,bas3,aas4,bas4]	= calc_threshold_and_params_in_size_dist(i_rhor_now,i_Frim_now,iconData["qni"][0,lev],iconData["qi"][0,lev],Frim_now,rho_rim_now,-99,nbins)#the last number is the number of bins for the calculation of the aas4 vector which is later stored in a lookup table
				#print 'time after calc_threshold_and_params_in_size_dist',time.time()-time0
				#write thresholds separating m-D relationships in one vector (dimension: level)
				dcrits_col[lev]	=	dcrits #separating nonspherical dense ice and graupel
				dcritr_col[lev]	=	dcritr #separating graupel and partially-rimed ice

				#copy m-D relationship coefficents to array with all heights

				cgp_col[lev]	=	cgp_now #a for graupel

				csr_col[lev]	=	csr	#a for partially-rimed ice
				#copy A-D relationship coefficents to array with all heights
				aas4_col[lev,:]   =	aas4    #aas for partially-rimed ice

			if (iconData["qi"][0,lev]>1e-10) and iconData["qni"][0,lev]>1 and debugging: #print parameter for debugging
				print 'level: '	+ str(lev) + 'timestep: ' + str(timecount)
				print 'qi: '	+ str(iconData["qi"][0,lev])
				print 'Frim: ' 	+ str(Frim_now)
				print 'i_Frim: '+ str(i_Frim_now)
				print 'rho_rim: '+ str(rho_rim_now)
				print 'i_rhor: '+ str(i_rhor_now)
				print 'cs1: ' 	+ str(cs1)
				print 'ds1: ' 	+ str(ds1)
				print 'dcrit: '	+ str(dcrit)				
				print 'cs: ' 	+ str(cs)
				print 'ds: ' 	+ str(ds)
				print 'dcrits: '+ str(dcrits)				
				print 'cgp: ' 	+ str(cgp_now)
				print 'dg: ' 	+ str(dg)
				print 'dcritr: '+ str(dcritr)				
				print 'csr: ' 	+ str(csr)
				print 'dsr: ' 	+ str(dsr)
		
		#write mass mixing ratios and number concentration to pseudo categories
		#(iconData)	=	write_massmix_num_conc_to_pseudo_categories(iconData,num_lev)
		#write hydrometeors into one variable named hydro_cmpl for mass mixing ratios and hydro_num_cmpl for number
		#(hydro_cmpl,hydro_num_cmpl)= write_massmix_num_conc_to_pamtra_4D_var(iconData,num_lev)	
		# Generate PAMTRA data dictonary
		pamData = dict()
		## Copy data to PAMTRA dictonary
		pamData["press"]  = iconData["pres"][0,:] # Pressure
                pamData["press"][pamData["press"]<0] = 101325 #sometimes the interpolation fails for the lowest levels

		pamData["relhum"] = iconData["rh"][0,:]  # Relative Humidity
		pamData["timestamp"] = unixtime #iconData["time"]
		try: #if lat/lon_icon is given as a vecor choose cell
			pamData["lat"] = np.rad2deg(lat_icon[cell])
			pamData["lon"] = np.rad2deg(lon_icon[cell])
		except:
			pamData["lat"] = np.rad2deg(lat_icon)
			pamData["lon"] = np.rad2deg(lon_icon)
		#print 'lat,lon' + str(pamData["lat"]) + ' , ' + str(pamData["lon"])

		#pamData["lfrac"] = icon_frland[timeframe]
		#pamData["wind10u"] = iconData["u"][0,1]
		#pamData["wind10v"] = iconData["v"][0,1]
		pamData["hgt"] = height
		#print pamData["hgt"][:]; sys.exit(1)
		pamData["temp"] = iconData["temp"][0,:]
		pamData["hydro_q"] = np.zeros([1,num_lev,4]) #TODO: query number of categories (here 4) automatically
		#pamData["hydro_q"] = hydro_cmpl[:,:]
		pamData["hydro_n"] = np.zeros([1,num_lev,4]) #TODO: query number of categories (here 4) automatically 
		#pamData["groundtemp"] = iconData["t_g"][0]
		#print pamData
		# Add them to pamtra object and create profile
		pam.createProfile(**pamData)
		#set hydrometeor properties
		pam.df.dataFullSpec

		pam.nmlSet["hydro_fullspec"] = True
		pam.df.addFullSpectra()
		#print 'time before  i_hydromet-loop',time.time()-time0
		for i_hydromet in range(0,4):#loop over different "categories": cloud water and rain are real categories, the number of categories in the ice phase is the number of scattering-routines used in the ice phase
			#setup diameter arrays
			if i_hydromet<=1:
                            pam.df.dataFullSpec["d_bound_ds"][0,0,:,i_hydromet,:],dum = np.meshgrid(np.logspace(-12,np.log10(8.5e-3),nbins+1),np.arange(0,num_lev))#2D-grid dimension:(height,bins); matrix with diameters which is repeated N_height times
                        else:
                            pam.df.dataFullSpec["d_bound_ds"][0,0,:,i_hydromet,:],dum = np.meshgrid(np.logspace(-12,-1,nbins+1),np.arange(0,num_lev))#2D-grid dimension:(height,bins); matrix with diameters which is repeated N_height times
			pam.df.dataFullSpec["d_ds"][0,0,:,i_hydromet,:] = pam.df.dataFullSpec["d_bound_ds"][0,0,:,i_hydromet,:-1] + 0.5 * np.diff(pam.df.dataFullSpec["d_bound_ds"][0,0,:,i_hydromet,:])#center of the bins defined by d_bound_ds
			if i_hydromet==0:
				iconData["mixr_now"]=iconData["qc"]
				iconData["numc_now"]=iconData["qnc"]
				mu_now = np.minimum(np.maximum(iconData["mu_c"][0,:],2),15) #set limits to mu as in P3-code
				lam_now = iconData["lamc"][0,:] #np.minimum(np.maximum(iconData["lamc"][0,:],(mu_now+1)*2.5e4),(mu_now+1)*1e6)#set limits to lam as in P3-code
			elif i_hydromet==1:
				iconData["mixr_now"]=iconData["qr"]
				iconData["numc_now"]=iconData["qnr"]
				mu_now = np.minimum(np.maximum(iconData["mu_r"][0,:],0),8.282) #set limits to mu as in P3-code
				lam_now = iconData["lamr"][0,:] #np.minimum(np.maximum(iconData["lamr"][0,:],(mu_now+1)*200),(mu_now+1)*1e5) #set limits to lam as in P3-code

			elif i_hydromet==2 or i_hydromet==3:
				iconData["mixr_now"]=iconData["qi"]
				iconData["numc_now"]=iconData["qni"]
				lam_now = iconData["lami"][0,:]
				mu_now = np.minimum(np.maximum(0.076*(iconData["lami"][0,:]/100.)**0.8-2.,0),6) #calculate mu and set limits to mu as in P3-code
			#calculate intercept parameter
			N0 = iconData["numc_now"][0,:]*lam_now**(mu_now+1.)/gamma(mu_now+1.)
			#for i_h in range(0,num_lev):	#loop over vertical levels
			#	#convert from specific to absolute units #this should be sufficient here, because before we only dealt with fraction like qi/qni where 1/m-3 or 1/km doesnt matter
                        N0 = q2abs(N0[:],pamData["temp"][:],pamData["press"][:],pamData["relhum"][:],iconData["qc"][0,:]+iconData["qr"][0,:]+iconData["qi"][0,:])


			if debugging:
				print 'i_hydromet: ' + str(i_hydromet)
				print 'iconData["mixr_now"]' + str(iconData["mixr_now"]) + 'lam_now' + str(lam_now)+ 'mu_now'+ str(mu_now) + 'N0' + str(N0)
			#calculate spectral number density for all vertical levels
			for i_h in range(0,num_lev):	#loop over vertical levels
				
				if i_hydromet==0 or i_hydromet==1: #liquid phase
					if onlycat[i_hydromet]=='1' and iconData["numc_now"][0,i_h]>1 and iconData["mixr_now"][0,i_h]>10**-10:#just do this if this category is not switched off; process only pixel with num_conc>1 and mix_ratio
						pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:] = N0[i_h]*pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,:]**mu_now[i_h]*np.exp(-lam_now[i_h]*pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,:])*np.diff(pam.df.dataFullSpec["d_bound_ds"][0,0,i_h,i_hydromet,:]) #not normalized number density [m-3]

						if False: #debugging and any(pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:]>1e-200):
							print 'i_h: ' + str(i_h) + ' pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:]' + str(pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:])
                                                
				#elif onlycat[i_hydromet]=='1'  and iconData["numc_now"][0,i_h]>10**-5 and iconData["mixr_now"][0,i_h]>10**-20: #ice phase
				#	pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:] = (N0[i_h]*pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,:]**mu_now[i_h])*np.exp(-lam_now[i_h]*pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,:])*np.diff(pam.df.dataFullSpec["d_bound_ds"][0,0,i_h,i_hydromet,:]) #not normalized number density [m-3]

			if True: #this is now used for warm and cold phase (because area and mass must be given also for liquid phase if higher moments or the full spectra should be calculated)
				#coefficients of m-D and A-D relationships
				#dense nonspherical ice 
				cs =	0.0121; ds = 1.9 #m-D parameters for dense nonspherical ice
				dcrit = (np.pi/(6.*cs)*900.)**(1./(ds-3.)) #lower limit of 
				bas2=1.88;aas2=0.2285*100.**bas2/(100.**2) #A-D parameters for dense nonspherical ice
		
				dsr = ds #m-D parameters for partially rimed ice
				bas4=1.88 #A-D parameters for partially rimed ice
		
				#calculate variables which are variable over the size range
				for i_h in range(0,num_lev):	#loop over vertical levels
					#calculate number density spectrum
					if iconData["numc_now"][0,i_h]>1 and iconData["mixr_now"][0,i_h]>10**-10:#just do this if this category is not switched off; process only pixel with num_conc>1 and mix_ratio
						pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:] = N0[i_h]*pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,:]**mu_now[i_h]*np.exp(-lam_now[i_h]*pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,:])*np.diff(pam.df.dataFullSpec["d_bound_ds"][0,0,i_h,i_hydromet,:]) #not normalized number density [m-3]
						
						if False: #debugging:
								print 'i_h: ' + str(i_h) + 'pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:]' + str(pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:])
		
					if any(pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:]>0): #( iconData["qi"][0,i_h]>10**(-10) ) and ( iconData["qni"][0,i_h]>1 ): #calculate only if N0 is not zero
						if debugging:
							print 'dcrit: ' + str(dcrit) + ' dcrits: ' + str(dcrits_col[i_h]) + ' dcritr: ' + str(dcritr_col[i_h])
						for i_bin in range(0,nbins):	#loop over size range
                                                        if i_hydromet==0 and onlycat[0]=='1': #cloud droplets #if you want to calculate higher moments or the full spectra this must also be defined
                                                                pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/4. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**2
                                                                pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/6. *1000. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**3
                                                                pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = 1000.
                                                                pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
                                                        elif i_hydromet==1 and onlycat[1]=='1': #rain
                                                                pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/4. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**2
                                                                pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/6. *1000. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**3
                                                                pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = 1000.
                                                                pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
                                                                
                                                        elif (pam.df.dataFullSpec["d_ds"][0,0,0,i_hydromet,i_bin]<=dcrit) and onlycat[2]=='1' and i_hydromet==2: #spherical ice
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] =  917 #taken from Morrison & Millbrandt (2015)
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/4. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**2
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/6. *pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin]*pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**3 
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
							elif (dcrit < pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] <= dcrits_col[i_h]) and onlycat[3]=='1' and i_hydromet==3: #dense nonspherical ice
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = 917 #not needed density is defined over cs and ds
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = aas2 * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**bas2
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = cs * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**ds  #emnp.pirical m-D-relation from Morrison & Millbrandt (2015)
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  0.6
							elif (max(dcrits_col[i_h],dcrit) < pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] <= dcritr_col[i_h]) and onlycat[4]=='1' and i_hydromet==2: #graupel #for Frim<0.3 also dcrit>dcrits is possible
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = cgp_col[i_h]*(6./np.pi)
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/4. * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] ** 2 
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = cgp_col[i_h] * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] ** 3 
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
							elif (dcritr_col[i_h] < pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]) and onlycat[5]=='1' and i_hydromet==3: #partially rimed
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = csr_col[i_h]*(6./np.pi)
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = aas4_col[i_h,i_bin] * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] ** bas4
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = csr_col[i_h] * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] ** dsr 
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  0.6
							else: #set variables to zero if onlycat[..]=0 lead you here
								pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,i_bin] = 0
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = 0
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = 0
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = 0
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
                                                        '''
                                                        if i_hydromet==0 and onlycat[0]=='1': #cloud droplets #if you want to calculate higher moments or the full spectra this must also be defined
                                                                pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/4. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**2
                                                                pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/6. *1000. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**3
                                                                pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = 1000.
                                                                pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
                                                        if i_hydromet==1 and onlycat[1]=='1': #rain
                                                                pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/4. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**2
                                                                pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/6. *1000. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**3
                                                                pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = 1000.
                                                                pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
							if (pam.df.dataFullSpec["d_ds"][0,0,0,i_hydromet,i_bin]<=dcrit) and onlycat[2]=='1' and i_hydromet==2: #spherical ice
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] =  917 #taken from Morrison & Millbrandt (2015)
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/4. *  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**2
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/6. *pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin]**3  pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**3 
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
							elif (dcrit < pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] <= dcrits_col[i_h]) and onlycat[3]=='1' and i_hydromet==3: #dense nonspherical ice
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = 917 #not needed density is defined over cs and ds
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = aas2 * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**bas2
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = cs * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]**ds  #emnp.pirical m-D-relation from Morrison & Millbrandt (2015)
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  0.6
							elif (max(dcrits_col[i_h],dcrit) < pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] <= dcritr_col[i_h]) and onlycat[4]=='1' and i_hydromet==2: #graupel #for Frim<0.3 also dcrit>dcrits is possible
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = cgp_col[i_h]*(6./np.pi)
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = np.pi/4. * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] ** 2 
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = cgp_col[i_h] * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] ** 3 
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
							elif (dcritr_col[i_h] < pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin]) and onlycat[5]=='1' and i_hydromet==3: #partially rimed
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = csr_col[i_h]*(6./np.pi)
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = aas4_col[i_h,i_bin] * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] ** bas4
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = csr_col[i_h] * pam.df.dataFullSpec["d_ds"][0,0,i_h,i_hydromet,i_bin] ** dsr 
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  0.6
							else: #set variables to zero if onlycat[..]=0 lead you here
								pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,i_bin] = 0
								pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,i_bin] = 0
								pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,i_bin] = 0
								pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,i_bin] = 0
								pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,i_bin] =  1.0
                                                        '''
					if False: #debugging:
						print 'i_h: ' + str(i_h) + 'pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:]' + str(pam.df.dataFullSpec["n_ds"][0,0,i_h,i_hydromet,:])
						print 'i_h: ' + str(i_h) + 'pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,:]' + str(pam.df.dataFullSpec["mass_ds"][0,0,i_h,i_hydromet,:])
						print 'i_h: ' + str(i_h) + 'pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,:]' + str(pam.df.dataFullSpec["area_ds"][0,0,i_h,i_hydromet,:])
						print 'i_h: ' + str(i_h) + 'pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,:]' + str(pam.df.dataFullSpec["rho_ds"][0,0,i_h,i_hydromet,:])
						print 'i_h: ' + str(i_h) + 'pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,:]' + str(pam.df.dataFullSpec["as_ratio"][0,0,i_h,i_hydromet,:])
                                #if i_hydromet==1:
                                #    from IPython.core.debugger import Tracer ; Tracer()()
		#print 'time before  runPamtra',time.time()-time0
		# Execute PAMTRA for MIRA 35 GHz
		#print "runPamtra"
		pam.runPamtra(rad_freq)
		#print 'time after  runPamtra',time.time()-time0
		# Write output to NetCDF4 file
		pam.writeResultsToNetCDF('output/example_ICON_radar_P3_'+ str(filecount) + "%02d" % (timecounter) + '.nc')
		timecounter	+=	1
		
	filecounter	+=	1	
	fh_ICON.close()
