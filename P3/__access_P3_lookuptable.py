'''
these functions accesses the p3 lookup table
'''
from IPython.core.debugger import Tracer ; debug = Tracer()
#load modules necessary for reading data
import numpy as np
from scipy.special import gamma
import sys
import os
import pandas as pd
import pprint as pprint
import time
def read_and_access_p3lookup_table(iconData,outvarlist=''):
	#iconData must contain : 'qi','qirim','qni','rime_dens'
	#list for variables which should be read/interpolated from lookup-table
	#outvarlist=['u_ns','p3mw_vi'] #mass weighted fall speed
	var_indices = [] #np.empty(len(outvarlist))

	#time0 = time.time() #start timing
	for i_var in range(0,len(outvarlist)):
		if outvarlist[i_var]=='p3nw_vi':
			var_indices.append(4)
		if outvarlist[i_var]=='p3mw_vi':
			var_indices.append(5)
		if outvarlist[i_var]=='aggr_ice':
			var_indices.append(6)
		if outvarlist[i_var]=='nlarge':
			var_indices.append(10)
		if outvarlist[i_var]=='nsmall':
			var_indices.append(11)
		if outvarlist[i_var]=='lami':
			var_indices.append(12)
		if outvarlist[i_var]=='p3dmean_ice':
			var_indices.append(15)
		if outvarlist[i_var]=='p3zw_vi':
			var_indices.append(17)

	if isinstance(iconData['rime_dens'],int) or isinstance(iconData['rime_dens'],float): #if iconData['rime_dens'] is just one number -> make it to a numpy array
		iconData['rime_dens']= np.array(iconData['rime_dens'])
	#avoid dividing by 0 #if birim is 0 'qirim' also should be 0
	#iconData['qirim'][iconData['birim']==0]=np.inf
	#iconData['qirim'][np.isinf(iconData['qirim']) | iconData['qirim']<0]=0
	#iconData['qi'][np.isinf(iconData['qi']) | iconData['qi']<0]=0
	#iconData['rime_dens']=np.maximum(np.minimum(iconData['qirim']/iconData['birim'],900),50)
	#print 'iconData[rime_dens] ' + str(iconData['rime_dens'])
	iconData['qnormi']=(iconData['qi'])/iconData['qni'] #qi here already is qi=qi+qirim!!
	#print 'iconData[qnormi]',iconData['qnormi']

	#allocate matrizes for reading the lookup-table
	lookuptable_matrix_numpy=np.zeros([5,4,50,len(outvarlist)+3])#17]) #rime_dens_index,rime_frac_index,norm_mix_ratio,variables
	#lookuptable_matrix_coll_numpy=np.zeros([5,3,1500,8])
	#define which columns should be read in
	basic_vars = [0,1,2] #always read the first three lines which show the position within the lookup table according to rime density, rime fraction and normalized mixing ratio
	read_cols = basic_vars + var_indices #read additionally the variables you want to interpolate
	######################################################
	##read in relevant part of lookup-table###############
	######################################################
	#get position in lookup-table (with information on the nearest neighbor and the relative distance to them)
	#if iconData['qni']>1e-10: #should not be necessary
	float_index_qnorm = (np.log10((iconData['qi'])/iconData['qni'])+18)/(0.1*np.log10(261.7))-11.
	#else:
	#	print 'Error: to low value of iconData[qni] in access_P3_lookuptable'
	float_index_Frim  = np.array((iconData['qirim']/iconData['qi'])*3.)
	try:
		float_index_rime_dens = np.zeros(iconData['qi'].shape)
		if np.any(iconData['rime_dens']<650):
			float_index_rime_dens[iconData['rime_dens']<650] = (iconData['rime_dens'][iconData['rime_dens']<650]-50.)*0.005
		if np.any(iconData['rime_dens']>=650):
			float_index_rime_dens[iconData['rime_dens']>=650] = (iconData['rime_dens'][iconData['rime_dens']>=650]-650.)*0.004 + 3.
	except: #in case iconData['qi'] is a float instead of an array	
		float_index_rime_dens = np.array([0])
		if iconData['rime_dens']<650:
			float_index_rime_dens = (iconData['rime_dens']-50.)*0.005
		if iconData['rime_dens']>=650:
			float_index_rime_dens = (iconData['rime_dens']-650.)*0.004 + 3.

	lookup=True #for speed up read from numpy instead of .dat which has a lot of entries we dont need
	if lookup:
		npzfile = np.load('p3_lookup_table_1-b08.npz')
		lookuptable_matrix_numpy=npzfile['lookuptable_matrix_numpy'][:,:,:,read_cols]#crop to read only what is needed
	else:
		save_lookup_table2numpy()
	#print 'after rime_dens_index loop',time.time()-time0

	def interpolating_lookuptable_variables(lookuptable_matrix_numpy,norm_index_rime_dens,norm_index_Frim,norm_index_qnorm,weighting_dimension,norm_interpolate_weight,start_position=[0,0,0]):
		#weighting_dimension is the dimension in which the interpolation is executed
		#start_position 1 means the next lower value in the lookup-table; 2 the next higher value in the lookup-table

		if weighting_dimension==0:
			lower_interpol_value = lookuptable_matrix_numpy[norm_index_rime_dens+start_position[0],norm_index_Frim+start_position[1],norm_index_qnorm,i_var+3]
			higher_interpol_value = lookuptable_matrix_numpy[norm_index_rime_dens+start_position[0],norm_index_Frim+start_position[1],norm_index_qnorm+1,i_var+3]
		elif weighting_dimension==1:
			lower_interpol_value = lookuptable_matrix_numpy[norm_index_rime_dens+start_position[0],norm_index_Frim,norm_index_qnorm+start_position[2],i_var+3]
			higher_interpol_value = lookuptable_matrix_numpy[norm_index_rime_dens+start_position[0],norm_index_Frim+1,norm_index_qnorm+start_position[2],i_var+3]
		elif weighting_dimension==2:
			lower_interpol_value = lookuptable_matrix_numpy[norm_index_rime_dens+start_position[0],norm_index_Frim+start_position[1],norm_index_qnorm+start_position[2],i_var+3]
			higher_interpol_value = lookuptable_matrix_numpy[norm_index_rime_dens+start_position[0],norm_index_Frim+start_position[1],norm_index_qnorm+start_position[2]+1,i_var+3]
		return lower_interpol_value + norm_interpolate_weight * (higher_interpol_value-lower_interpol_value)
	######################################################
	###interpolate between the lookup-table values########
	######################################################
	for i_var in range(0,len(outvarlist)):
		#getting position and weighting between next lower and next higher value in the lookup-table for rime density
		norm_index_rime_dens = np.maximum(np.minimum(float_index_rime_dens.astype(int),3),0) #next lower value in the lookup-table
		norm_interpolate_weight_rime_dens=float_index_rime_dens-norm_index_rime_dens #weighting between next lower and next higher value in the lookup-table
		#getting position and weighting between next lower and next higher value in the lookup-table for rime fraction
		norm_index_Frim = np.maximum(np.minimum(float_index_Frim.astype(int),2),0) #next lower value in the lookup-table
		norm_interpolate_weight_Frim=float_index_Frim-norm_index_Frim #weighting between next lower and next higher value in the lookup-table
		#getting position and weighting between next lower and next higher value in the lookup-table for normalized mass mixing ratio
		norm_index_qnorm = np.minimum(float_index_qnorm.astype(int),48) #next lower value in the lookup-table
		norm_interpolate_weight_qnorm=float_index_qnorm-norm_index_qnorm #weighting between next lower and next higher value in the lookup-table
		
		#print 'norm_index_rime_dens',norm_index_rime_dens,'norm_interpolate_weight_rime_dens',norm_interpolate_weight_rime_dens,'norm_index_Frim',norm_index_Frim,'norm_interpolate_weight_Frim',norm_interpolate_weight_Frim,'norm_index_qnorm',norm_index_qnorm,'norm_interpolate_weight_qnorm',norm_interpolate_weight_qnorm

		###interpolate into the dimension of the normalized mass mixing ratio first then into the rimed fraction and at last into the rimed density
		#beginning from the point in the lookuptable_matrix_numpy[norm_index_rime_dens,norm_index_Frim,:][norm_index_qnorm] which represents the next lower value in the lookup-table for all three determining variables
		qnorm_interpol_rimdens0_frim0 = interpolating_lookuptable_variables(lookuptable_matrix_numpy,norm_index_rime_dens,norm_index_Frim,norm_index_qnorm,2,norm_interpolate_weight_qnorm,[0,0,0])
		qnorm_interpol_rimdens0_frim1 = interpolating_lookuptable_variables(lookuptable_matrix_numpy,norm_index_rime_dens,norm_index_Frim,norm_index_qnorm,2,norm_interpolate_weight_qnorm,[0,1,0])
		#interpolate those two values with the weight of the rimed fraction
		frim_qnorm_interpol_rimdens0 = qnorm_interpol_rimdens0_frim0 + norm_interpolate_weight_Frim * (qnorm_interpol_rimdens0_frim1-qnorm_interpol_rimdens0_frim0)

		#beginning from the point in the lookuptable_matrix_numpy[norm_index_rime_dens+1,norm_index_Frim,:][norm_index_qnorm] which represents the next lower value in the lookup-table for all three determining variables
		qnorm_interpol_rimdens1_frim0 = interpolating_lookuptable_variables(lookuptable_matrix_numpy,norm_index_rime_dens,norm_index_Frim,norm_index_qnorm,2,norm_interpolate_weight_qnorm,[1,0,0])
		qnorm_interpol_rimdens1_frim1 = interpolating_lookuptable_variables(lookuptable_matrix_numpy,norm_index_rime_dens,norm_index_Frim,norm_index_qnorm,2,norm_interpolate_weight_qnorm,[1,1,0])
		#interpolate those two values with the weight of the rimed fraction
		frim_qnorm_interpol_rimdens1  = qnorm_interpol_rimdens1_frim0 + norm_interpolate_weight_Frim * (qnorm_interpol_rimdens1_frim1-qnorm_interpol_rimdens1_frim0)

		#final interpolation between frim_qnorm_interpol_rimdens0 and frim_qnorm_interpol_rimdens1
		iconData[outvarlist[i_var]]   = (frim_qnorm_interpol_rimdens0 + norm_interpolate_weight_rime_dens * (frim_qnorm_interpol_rimdens1-frim_qnorm_interpol_rimdens0))
	return iconData


def save_lookup_table2numpy():
	lookuptable_matrix_numpy=np.zeros([5,4,50,17]) #rime_dens_index,rime_frac_index,norm_mix_ratio,variables
	for rime_dens_index in range(0,5):
		for rime_frac_index in range(0,4):
			#lookuptable_matrix = pd.read_table('output/p3_lookup_table_1.dat',nrows=50,delimiter='    ',skiprows=(rime_dens_index*4+rime_frac_index)*1550,usecols=read_cols,header=None)
			lookuptable_matrix = pd.read_table('output/p3_lookup_table_1-b08.dat',nrows=50,skiprows=(rime_dens_index*4+rime_frac_index)*1550,header=None, engine='python',delim_whitespace=True)

			#store the read in table in a numpy vector		
			lookuptable_matrix_numpy[rime_dens_index,rime_frac_index,:,:]=lookuptable_matrix.values
	np.savez('p3_lookup_table_1-b08.npz',lookuptable_matrix_numpy=lookuptable_matrix_numpy)
