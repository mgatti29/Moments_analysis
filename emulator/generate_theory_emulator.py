#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import ctypes
from ctypes import *
import sys
import numpy as np
import os
import numpy.linalg as LA
import time
import timeit
import pickle
from multiprocessing import Pool
from contextlib import closing
import pandas as pd
from tqdm import tqdm
import h5py
from multiprocessing import *
from mpi4py import MPI 
import gc

import faulthandler; faulthandler.enable()

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(name):
    
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')

try:
    import queue
except ImportError:
    import Queue as queue

'''
This code produces MASKED <d^2>(z) and <d^3>(z) predictions for a grid of different cosmological inputs.
This is needed for the emulator. You can use different mixing matrixes - M_EE is the default for shear moments.
Although this code want redshift distributions as input, they don't need to be realistic - this code only generates
the density moments as a function of redshift before integrating them along the redshift direction. the full computation
of the shear / density moments integrated along the lin of sight with the shear /n(z) kernels is done by a python code after
the emulator has been trained.

The code needs to be run twice, with (NL_p = 1) and without (NL_p = 0) the small-scales fitting formulae.


to run, it might require export HDF5_USE_FILE_LOCKING=FALSE


srun --nodes=1 --tasks-per-node=3 --cpus-per-task=20 --cpu-bind=cores --mem=110GB python generate_theory_emulator.py
'''


####################################################
#                 INPUT parameters
####################################################

# these values for the cosmological parameters are ignored here.
# it loops over the emulator points later.
# The nuisancwe parameters are set to 0. The marginalisation over them will happen in the python code.

# cosmological parameters **************************
Omega_m = 0.279 
Omega_b = 0.046 
sigma_8 = 0.82 
n_s = 0.97 
h_100 = 0.7
# IA parameters ************************************
A0 = 0.
z0 = 0.63
alpha0 = 0.
# redshift parameters ******************************
z_shift = np.zeros(5).astype(np.double)
z_spread =  np.ones(5).astype(np.double)

# OWL parameters ***********************************
# 0: no effect. 1: owl ; 2: window + owl; 3: windowed
baryonic_effect = 0
r_OWL = 0.0
dirname = "./owls/"
powz = os.path.join(dirname, 'powtable_z.txt')
powk = os.path.join(dirname, 'powtable_k.txt')
DM_FILENAME = os.path.join(dirname, 'powtable_DMONLY_all_P.txt')
UPPER_FILENAME = os.path.join(dirname, 'powtable_NOSN_all_P.txt')
LOWER_FILENAME = os.path.join(dirname, 'powtable_AGN_all_P.txt')
elle_filename = os.path.join(dirname, 'powelle.txt')

# NON LINEAR FITTING FORMULAE ************
# 0: standard PT; 1 SC formulae; 2 GM formulae. 1 is Default.
NL_p = 1

# smoothing scales *********************************
smooth = np.array([3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.])
nn = "./theory_code/angles.txt" # the code will read from this file the smoothing scales.

# *************************************************
fract_area = 1./8.66# needed for third moment

# Redshift distributions **************************
# format label: name_file + bin[i] + .txt
# format files: first column z, second n(z)
name_file = '/global/project/projectdirs/des/mgatti/Moments_analysis/Nz/DESY3_unblind_'
red_bins = ['1', '2', '3', '4']

# bins to consider ********************************

bins_array_2 = [[3,3]]
bins_array_3 = [[3,3,3]]


# 1 for lensing, 0 for galaxies.
# it is actually irrelevant for this code.

lensing_A = 1  
lensing_B = 1

# OUTPUT FOLDER(S) *********************************
label_output = ''
# folder where to save the outputs
outp_folder = '/global/project/projectdirs/des/mgatti/Moments_analysis/EMU_sc/'
# file (pkl) containing the training points.
params_train ="/global/project/projectdirs/des/mgatti/Moments_analysis/emulator_training_points/points_3000_parameters_5"


# MIXING MATRIX OPTIONS  ***************************
folder_mask ='/global/project/projectdirs/des/mgatti/Moments_analysis/'
name_mask1 = 'mode_coupling_matrix_2048_1024'
type_mask = 'ME' # you can choose between ME, MB, MgE, MEE
len_mask = 2048 # set to 0 if no mask has to be applied
mask_depending_on_theta = False # set to false (I used set to True to generate the emulator as a function of smoothing scales.)


# PIXEL WINDOW FUNCTION *****************************
name_Cl_pix_mask = './theory_code/Cl_pix/CL_pix1_2048_1024.txt'
len_pix_mask = 2048

####################################################
#                   CODE 
####################################################

if __name__ == '__main__':


    # it saves the smoothing_scales in the 'nn' file.
    fil1 = open(nn,"w")
    for i in range(len(smooth)):
        fil1.write('{0}\n'.format(smooth[i]))
    fil1.close()


    # it saves some configurations *********************
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


    # creates the output folder if it doesn't exist
    if not os.path.exists(outp_folder):
        os.mkdir(outp_folder)

    # conversion of a few variables into the format needed by the c++ code
    zsf = (z_shift).ctypes.data_as(POINTER(c_double))
    zsp = (z_spread).ctypes.data_as(POINTER(c_double))

    bins_array_2_o = np.array(bins_array_2)
    bins_array_3_o = np.array(bins_array_3)
    bins_array_2 = bins_array_2_o.flatten()
    bins_array_3 = bins_array_3_o.flatten()

    pofz_file1 = []
    for bin in red_bins:
        pofz_file1.append(ctypes.c_char_p((name_file + bin + '.txt').encode('utf-8')))

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
            else:
                name_file = folder_mask+name_mask1+'.h5'
                h5f = h5py.File(name_file, 'r')
                Memute = h5f[type_mask][:]
                ME[i, :] = Memute
    else:
        ME = np.ones((len(smooth),11), dtype=float)

    ME = tuple(np.ascontiguousarray(ME.ravel()))
    type_matrix   = (ctypes.c_double * len(ME))
    mask = type_matrix (*ME)

    end = timeit.default_timer()
    tt = end-start
    start0 = timeit.default_timer()
    print ("matrix loading : {0:2.2f} s ".format(end-start))



    #***************************************************
    if len_pix_mask!=0:
        Clpix = np.ones(10001, dtype=float)
        Clpix2 = np.array(np.loadtxt(name_Cl_pix_mask), dtype=float)
        for i in range(len(Clpix2)):
            Clpix[i] = Clpix2[i]
    else:
        Clpix = np.ones(10001, dtype=float)

    # ***********************




    start = timeit.default_timer()


    def make_output_list1(bins_array_2, bins_array_3, lensing_A, lensing_B, name_mask,type_mask, sigma_8, o_m,o_b,h100,ns):
        '''
        it gives back the output format for the moments prediction.
        '''
        save_out_list = []
        for bin in bins_array_2:
            if len_mask != 0:
                xx = outp_folder + '/'+label_output+'Buzzard_f{0}f{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}.txt'.format(lensing_A, lensing_B, bin[0], bin[1], '',
                                                                             '',sigma_8, o_m,o_b,h100,ns,type_mask, name_mask)
                # print xx
                save_out_list.append(ctypes.c_char_p(xx.encode('utf-8')))
            else:
                xx = outp_folder + '/'+label_output+'Buzzard_f{0}f{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}.txt'.format(lensing_A, lensing_B, bin[0], bin[1], '',
                                                                             '',sigma_8, o_m,o_b,h100,ns)
                # print xx
                save_out_list.append(ctypes.c_char_p(xx.encode('utf-8')))
        for bin in bins_array_3:
            if len_mask != 0:
                xx = outp_folder + '/'+label_output+'Buzzard_f{0}f{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}.txt'.format(lensing_A, lensing_B, bin[0], bin[1],
                                                                                 bin[2], '','',sigma_8, o_m,o_b,h100,ns,type_mask, name_mask)
                # print xx
                save_out_list.append(ctypes.c_char_p(xx.encode('utf-8')))
            else:
                xx = outp_folder + '/'+label_output+'Buzzard_f{0}f{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}.txt'.format(lensing_A, lensing_B, bin[0], bin[1],
                                                                                 bin[2], '','',sigma_8, o_m,o_b,h100,ns)
                # print xx
                save_out_list.append(ctypes.c_char_p(xx.encode('utf-8')))
        select_type2 = (c_char_p * len(save_out_list))
        save_out_list2 = select_type2()
        for key, item in enumerate(save_out_list):
            save_out_list2[key] = item
        return save_out_list2, select_type2




    import signal
    def timeout_handler(num, stack):
        print("Received SIGALRM")
        raise Exception("FUBAR")
    signal.signal(signal.SIGALRM, timeout_handler)

    def run_it(i, vvv):
        # eumator training point values: (omegam, sigma8, ...)
        vv = vvv[i]

        # updates cosmologicval parameter values
        Omega_m = vv[0]
        sigma_8 = vv[2]
        Omega_b = vv[1]
        n_s = vv[4]
        h_100 = vv[3]

        
        
        save_out_list2, select_type2 = make_output_list1(bins_array_2_o, bins_array_3_o, lensing_A, lensing_B,
                                                        name_mask1, type_mask, sigma_8, Omega_m, Omega_b, n_s, h_100)
        #try:
            
        if (i!=2254) and (i!=2228) :

            # these are the names of the key quantities <d^2>(z) and <d^3>(z) we need to save
            #if (not os.path.exists(outp_folder + 'd2_{0}_{1}_{2}_{3}_{4}_{5}'.format(type_mask,sigma_8, Omega_m, Omega_b, n_s, h_100))) or (not os.path.exists(outp_folder + 'd3_{0}_{1}_{2}_{3}_{4}_{5}'.format(type_mask,sigma_8, Omega_m, Omega_b, n_s, h_100))) :
                d_outp = [(outp_folder + 'd2_{0}_{1}_{2}_{3}_{4}_{5}'.format(type_mask,sigma_8, Omega_m, Omega_b, n_s, h_100)).encode('utf-8'),
                     ( outp_folder + 'd3_{0}_{1}_{2}_{3}_{4}_{5}'.format(type_mask,sigma_8, Omega_m, Omega_b, n_s, h_100)).encode('utf-8')]
                d_outp_type = (c_char_p * len(d_outp))
                d_outp2 = d_outp_type()
                for key, item in enumerate(d_outp):
                    d_outp2[key] = item

                print ('cosmo parameters')
                print ( 'Om: {0:2.2f}'.format(Omega_m))
                print ( 'Ob: {0:2.2f}'.format(Omega_b))
                print ( 's8: {0:2.2f}'.format(sigma_8))
                print ( 'ns: {0:2.2f}'.format(n_s))
                print ( 'h100: {0:2.2f}'.format(h_100))

                print_moments = lib.print_moments
                
                print_moments.argtypes = [ctypes.c_char_p, POINTER(c_double), ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, ctypes.c_double, ctypes.c_double, select_type, ctypes.c_int,
                                      POINTER(c_long), ctypes.c_int, POINTER(c_long), ctypes.c_int,
                                      select_type2, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, POINTER(c_double), POINTER(c_double), type_matrix, d_outp_type,
                                      ctypes.c_int, ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
                print_moments.restype = POINTER(c_double)
                print_moments(nn.encode('utf-8'), smooth.astype(np.double).ctypes.data_as(POINTER(c_double)), len(smooth),
                          Omega_m, Omega_b, sigma_8, n_s, h_100,
                          select, len(pofz_file1), bins_array_2.astype(np.long).ctypes.data_as(POINTER(c_long)),
                          int(len(bins_array_2) / 2), bins_array_3.astype(np.long).ctypes.data_as(POINTER(c_long)),
                          int(len(bins_array_3) / 3), save_out_list2, len_mask, name_Cl_pix_mask.encode('utf-8'), len_pix_mask, fract_area, lensing_A, lensing_B, 
                         'None'.encode('utf-8'),'None'.encode('utf-8'), A0, z0, alpha0, zsf, zsp, mask, d_outp2, baryonic_effect,r_OWL, 
                         DM_FILENAME.encode('utf-8'),UPPER_FILENAME.encode('utf-8') ,LOWER_FILENAME.encode('utf-8'),powz.encode('utf-8'),powk.encode('utf-8'),elle_filename.encode('utf-8'), NL_p)
                del print_moments
                gc.collect()
                print ('Done!')

        #except Exception as ex:
        #    if "FUBAR" in ex:
        #        print("Something went wrong with this cosmoslogy")
        #finally:
        #    signal.alarm(0)


    import os

    import math


    vvv = load_obj(params_train)


    sims = range(vvv.shape[0])

    # load the libraries of the c++ code.
    lib=ctypes.cdll.LoadLibrary("./theory_code/trough.so")
    #lib=ctypes.cdll.LoadLibrary("./trough.so")
    run_it(0,vvv)
    run_count =  0
    #while run_count<vvv.shape[0]:
    #    comm = MPI.COMM_WORLD
    #    print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
    #    try:
    #        run_it(run_count+comm.rank,vvv)
    #    except:
    #        pass
    #    run_count+=comm.size
    #    comm.bcast(run_count,root = 0)
    #    comm.Barrier() 
    

