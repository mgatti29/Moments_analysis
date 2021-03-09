#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
#import emcee
#import scipy.optimize as op
import ctypes
from ctypes import *
import sys
import numpy as np
import os
import numpy.linalg as LA
import time
import timeit
import pickle
#import corner
from multiprocessing import Pool
from contextlib import closing
#import yaml
import pandas as pd
#from tqdm import tqdm
import h5py

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


lib=ctypes.cdll.LoadLibrary("./trough.so")

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute = pickle.load(f)
        f.close()
    return mute





####################################################
#                       INPUT
####################################################
# so far we're in the fast version. S3 computed without the Matrix and with NL PS.
# cosmology ***************************
aaa = 0



#Picola
#Omega_m = 0.31 #+ aaa * 0.069
#Omega_b = 0.048 #+ aaa * 0.0042
#sigma_8 = 0.83 #+ aaa * .0725
#n_s = 0.96 #+ aaa * 0.14309020171
#h_100 = 0.69 #+ aaa * 0.054



#Buzzard
Omega_m = 0.286 #+ aaa * 0.069
Omega_b = 0.047 #+ aaa * 0.0042
sigma_8 = 0.82 #+ aaa * .0725
n_s = 0.96 #+ aaa * 0.14309020171
h_100 = 0.7 #+ aaa * 0.054



#Taka
Omega_m = 0.279 #+ aaa * 0.069
Omega_b = 0.046 #+ aaa * 0.0042
sigma_8 = 0.82 #+ aaa * .0725
n_s = 0.97 #+ aaa * 0.14309020171
h_100 = 0.7 #+ aaa * 0.054


nside = 1024


outp_folder = '/global/homes/m/mgatti/Mass_Mapping/emulator/theory_predictions/Taka_gm_GM_{0}_'.format(nside)

outp_folder = '/global/homes/m/mgatti/Mass_Mapping/emulator/theory_predictions/Taka_GM_new_nl_masked_{0}_'.format(nside)


# bins to compute *****************
# IA params
A0 = 0.#.#1.
z0 = 0.#62
alpha0 = 0.
z_shift = -1.*np.array([0.0063,0.0063,0.0063,0.0063,0.0063]).astype(np.double)
#z_shift = -np.array([ 0.0029973  , 0.00422811, -0.02563841, -0.11791389, -0.0147509 ]).astype(np.double) #shift to have y1 error shift
#z_shift = 0*np.array([-0.0189973,  -0.01722811 , 0.03663841 , 0.13991389  ,0.0247509 ]) #shift to have <z_t> = <z_bpz>
z_spread =  1.*np.array([1.,1.,1.,1.,1.]).astype(np.double)





baryonic_effect = 3 # with 2:  window + owl  # with 3 only windowd
r_OWL = 0.0
#theory_Y3_1024

label_output = ''
# format of the files : name_file_red_bins.txt (e.g.: multi_z_Buzzard_g1g2_0.2_0.43.txt)




#name_file = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/moment_computation_fast/multi_z_Buzzard_g1g2_'
name_file = "/global/homes/m/mgatti/Mass_Mapping/run_measurement/Buzzard_y3_nz/Buzzard_y3_z_"
#name_file = "/global/homes/m/mgatti/Mass_Mapping/run_measurement/Buzzard_y3_nz/Buzzard_y3_z_16_"

red_bins = ['0.2_0.43','0.43_0.63','0.63_0.9','0.9_1.3','0.2_1.3']



bins_array_2 = [[1,1],[2,2],[3,3],[4,4],[5,5],[4,3],[4,2],[4,1],[3,2],[3,1],[2,1]]
bins_array_3 = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[4,3,4],[4,2,4],[4,1,4],[3,2,3],[3,1,3],[2,1,2],[4,3,3],[4,2,2],[4,1,1],[3,2,2],[3,1,1],[2,1,1],[1,2,3],[1,2,4],[1,3,4],[2,3,4]]

#bins_array_2 = [[1,1],[2,2],[3,3],[4,4],[5,5]]
#bins_array_3 = [[5,5,5],[1,1,1],[2,2,2],[3,3,3],[4,4,4]]

lensing_A = 1  # 1 lensed, 0 matter
lensing_B = 1

# smoothing scales ***
smooth = np.array([0.0,2.0,3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.])
#smooth = np.array([3.2,8.2,21.0,54.,138.0])

fract_area = 1./0.12245750427246095 # for masking purposes
fract_area = 1./0.115
if nside==1024:
    folder_mask ='//global/cscratch1/sd/mgatti/Mass_Mapping/Maps_Y3//'.format(nside)
   
else:
    folder_mask ='/global/cscratch1/sd/mgatti/Mass_Mapping/Maps_Y3/'.format(nside)
    

name_mask1 = 'mode_coupling_matrix_{0}_{1}'.format(nside*2,nside)
type_mask = 'ME'
len_mask = 2048#nside*2# set to 0 if no mask has to be applied
mask_depending_on_theta = False




#name_Cl_pix_mask = 'CL_pix1_1024.txt'
name_Cl_pix_mask = './Cl_pix/CL_pix1_2048_1024.txt'
name_Cl_pix_mask = './Cl_pix/CL_pix1_1024_512.txt'
name_Cl_pix_mask = './Cl_pix/CL_pix1_{0}_{1}.txt'.format(nside*2,nside)
len_pix_mask = nside*2


# OWL baryonic effects
# 0 = False, 1 = True
dirname = "./owls/"


powz = os.path.join(dirname, 'powtable_z.txt')
powk = os.path.join(dirname, 'powtable_k.txt')

DM_FILENAME = os.path.join(dirname, 'powtable_DMONLY_all_P.txt')
UPPER_FILENAME = os.path.join(dirname, 'powtable_NOSN_all_P.txt')
LOWER_FILENAME = os.path.join(dirname, 'powtable_AGN_all_P.txt')
elle_filename = os.path.join(dirname, 'powelle.txt')



data_mcmc = '2_mom_flask_v.txt'
inv_cov_mcmc = '2_mom_buzz_inv_cov.txt'




rewrite = True
load_sampler_v= False



sampling_mcmc = False


input_conf = dict()
input_conf.update({'name_mask':name_mask1})
input_conf.update({'name_Cl_pix_mask':name_Cl_pix_mask})
input_conf.update({'len_pix_mask':len_pix_mask})
input_conf.update({'len_mask':len_mask})
input_conf.update({'fract_area':fract_area})
input_conf.update({'smooth':smooth})
input_conf.update({'bins_array_2':bins_array_2})
input_conf.update({'bins_array_3':bins_array_3})
input_conf.update({'name_file':name_file})
input_conf.update({'A0':A0})
input_conf.update({'z0':z0})
input_conf.update({'z_shift':z_shift})
input_conf.update({'z_spread ':z_spread})


#######################################################
#######################################################

zsf = (z_shift).ctypes.data_as(POINTER(c_double))
zsp = (z_spread).ctypes.data_as(POINTER(c_double))

if not os.path.exists(outp_folder):
    os.mkdir(outp_folder)

bins_array_2_o = np.array(bins_array_2)
bins_array_3_o = np.array(bins_array_3)
bins_array_2 = bins_array_2_o.flatten()
bins_array_3 = bins_array_3_o.flatten()


smooth = np.array(smooth)

pofz_file1 = []
for bin in red_bins:
    pofz_file1.append(ctypes.c_char_p(name_file + bin+'.txt'))

select_type = (c_char_p * len(pofz_file1))
select = select_type()
for key, item in enumerate(pofz_file1):
    select[key] = item


# READ mask files **********************************


name_mask_array = []
start = timeit.default_timer()
if len_mask!=0:
    ME = np.zeros((len(smooth),len_mask*len_mask))
    for i,sm in enumerate(smooth):
        if mask_depending_on_theta:
            name_file = folder_mask+name_mask1+'_{0}.h5'.format(sm)
            h5f = h5py.File(name_file, 'r')
            Memute = h5f[type_mask][:]
            ME[i,:] = Memute
            name_mask_array.append(folder_mask+name_mask1+'_'+str(sm)+'.txt')
        else:
            name_file = folder_mask+name_mask1+'.h5'
            h5f = h5py.File(name_file, 'r')
            Memute = h5f[type_mask][:]
            ME[i, :] = Memute
            name_mask_array.append(folder_mask+name_mask1 +'.txt')

else:
    ME = np.ones((len(smooth),11), dtype=float)
    for i, sm in enumerate(smooth):
        name_mask_array.append(folder_mask+name_mask1 + '.txt')


#ME = np.arange(10);#np.array(np.ones((5,2)))

ME= tuple(np.ascontiguousarray(ME.ravel()))

type_matrix   = (ctypes.c_double * len(ME))
a = type_matrix (*ME)

#ptr = ctypes.cast(ME, ctypes.POINTER(ctypes.c_double * len(ME)))
#a = np.asarray(ptr.contents)

end = timeit.default_timer()
print (len(ME))
tt = end-start
start0 = timeit.default_timer()
print ("matrix loading : {0:2.2f} s ".format(end-start))

# converting the matrix in a format the code can read: double**

mselect_type = (c_char_p * len(name_mask_array))
mselect = mselect_type()


for key, item in enumerate(name_mask_array):
    mselect[key] = item

#***************************************************
if len_pix_mask!=0:
    Clpix = np.ones(10001, dtype=float)
    Clpix2 = np.array(np.loadtxt(name_Cl_pix_mask), dtype=float)
    for i in range(len(Clpix2)):
        Clpix[i] = Clpix2[i]
    #Clpix2 =np.array(np.loadtxt(name_Cl_pix_mask), dtype=float)[3]

    #Clpix = 1.*np.array(np.loadtxt(name_Cl_pix_mask), dtype=float)
   # print Clpix[:1000]
else:
    Clpix = np.ones(10001, dtype=float)


#***************************************************
fil1 = open("angles.txt","w")
for i in range(len(smooth)):
    fil1.write('{0}\n'.format(smooth[i]))
fil1.close()
#if sampling_mcmc:
#    data_mcmc = np.array(np.loadtxt(name_data), dtype=float)
#    inv_cov_mcmc = np.array(np.loadtxt(name_inv_cov), dtype=float)

# ***********************

def make_output_list(bins_array_2, bins_array_3, lensing_A, lensing_B, name_mask,type_mask):
    save_out_list = []
    for bin in bins_array_2:
        if len_mask != 0:
            xx = outp_folder + '/'+label_output+'Buzzard_f{0}f{1}_{2}_{3}_{4}_{5}_{6}_{7}.txt'.format(lensing_A, lensing_B, bin[0], bin[1], '',
                                                                         '',type_mask, name_mask)
            # print xx
            save_out_list.append(ctypes.c_char_p(xx))
        else:
            xx = outp_folder + '/'+label_output+'Buzzard_f{0}f{1}_{2}_{3}_{4}_{5}.txt'.format(lensing_A, lensing_B, bin[0], bin[1], '',
                                                                         '')
            # print xx
            save_out_list.append(ctypes.c_char_p(xx))
    for bin in bins_array_3:
        if len_mask != 0:
            xx = outp_folder + '/'+label_output+'Buzzard_f{0}f{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.txt'.format(lensing_A, lensing_B, bin[0], bin[1],
                                                                             bin[2], '','',type_mask, name_mask)
            # print xx
            save_out_list.append(ctypes.c_char_p(xx))
        else:
            xx = outp_folder + '/'+label_output+'Buzzard_f{0}f{1}_{2}_{3}_{4}_{5}_{6}.txt'.format(lensing_A, lensing_B, bin[0], bin[1],
                                                                             bin[2], '','')
            # print xx
            save_out_list.append(ctypes.c_char_p(xx))
    select_type2 = (c_char_p * len(save_out_list))
    save_out_list2 = select_type2()
    for key, item in enumerate(save_out_list):
        save_out_list2[key] = item
    return save_out_list2, select_type2


print_moments = lib.print_moments

d_outp = [outp_folder + '/d2_{0}_{1}.txt'.format(sigma_8, Omega_m), outp_folder + '/d3_{0}_{1}.txt'.format(sigma_8, Omega_m)]
d_outp_type = (c_char_p * len(d_outp))
d_outp2 = d_outp_type()
for key, item in enumerate(d_outp):
    d_outp2[key] = item


#######################################################
#######################################################
start = timeit.default_timer()
if 1 == 1:


    if sampling_mcmc:
        pass

    else:

            nn = "angles.txt"
            print('Computing moments ****** \n')
            #print('kappa\n')


            save_out_list2,select_type2 = make_output_list(bins_array_2_o,bins_array_3_o,lensing_A,lensing_B,name_mask1,type_mask)

            print_moments.argtypes = [ctypes.c_char_p,POINTER(c_double), ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, ctypes.c_double, ctypes.c_double, select_type, ctypes.c_int,
                                      POINTER(c_long), ctypes.c_int, POINTER(c_long), ctypes.c_int,
                                      select_type2, mselect_type,mselect_type,
                                      ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, POINTER(c_double), POINTER(c_double),type_matrix,d_outp_type,
                                      ctypes.c_int,ctypes.c_double,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p]
            print_moments.restype = POINTER(c_double)





            print Omega_m, Omega_b, sigma_8, n_s, h_100,A0, z0, alpha0
            print_moments(nn,smooth.astype(np.double).ctypes.data_as(POINTER(c_double)), len(smooth),
                          Omega_m, Omega_b, sigma_8, n_s, h_100,
                          select, len(pofz_file1), bins_array_2.astype(np.long).ctypes.data_as(POINTER(c_long)),
                          int(len(bins_array_2) / 2), bins_array_3.astype(np.long).ctypes.data_as(POINTER(c_long)),
                          int(len(bins_array_3) / 3), save_out_list2,
                          mselect,mselect,
                          len_mask, name_Cl_pix_mask, len_pix_mask, fract_area, lensing_A, lensing_B, data_mcmc,
                          inv_cov_mcmc, A0, z0, alpha0, zsf, zsp,a,d_outp2, baryonic_effect,r_OWL,  DM_FILENAME,UPPER_FILENAME ,LOWER_FILENAME,powz,powk,elle_filename)#a.astype(np.double).ctypes.data_as(POINTER(c_double)))

            time_tot = timeit.default_timer() - start0
            input_conf.update({'time loading': tt})
            input_conf.update({'time moments': time_tot})

            if len_mask != 0:
                save_obj(outp_folder + '/' + label_output + '_'+type_mask+'_'+name_mask1+'_conf', input_conf)
            else:
                save_obj(outp_folder + '/' + label_output + '__conf', input_conf)
            #from ctypes import _pointer_type_cache

            #del _pointer_type_cache[a]
            #print('\ndelta\n')

