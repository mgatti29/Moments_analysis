#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import emcee
import scipy.optimize as op
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
import yaml
import pandas as pd
from tqdm import tqdm


def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# ***********************
def make_output_list(bins_array_2, bins_array_3, lensing_A, lensing_B,label_output):
    save_out_list = []
    for bin in bins_array_2:
        xx = outp_folder + '/' + label_output +'_f{0}f{1}_{2}_{3}'.format(lensing_A, lensing_B,bin[0],bin[1])
        save_out_list.append(ctypes.c_char_p(xx.encode('utf-8')))

    for bin in bins_array_3:
        xx = outp_folder + '/' + label_output + '_f{0}f{1}_{2}_{3}_{4}'.format(lensing_A, lensing_B, bin[0], bin[1], bin[2])
        save_out_list.append(ctypes.c_char_p(xx.encode('utf-8')))


    select_type2 = (c_char_p * len(save_out_list))
    save_out_list2 = select_type2()
    for key, item in enumerate(save_out_list):
        save_out_list2[key] = item
    return save_out_list2, select_type2



lib = ctypes.cdll.LoadLibrary("./trough.so")

####################################################
#                       INPUT
####################################################

# cosmology ***************************
aaa = 0
'''
Omega_m = 0.286  # + aaa * 0.069
Omega_b = 0.047  # + aaa * 0.0042
sigma_8 = 0.82  # + aaa * .0725
n_s = 0.96  # + aaa * 0.14309020171
h_100 = 0.7  # + aaa * 0.054

'''
#pcola
Omega_m = 0.31 #+ aaa * 0.069
Omega_b = 0.048 #+ aaa * 0.0042
sigma_8 = 0.83 #+ aaa * .0725
n_s = 0.96 #+ aaa * 0.14309020171
h_100 = 0.69 #+ aaa * 0.054


#taka
Omega_m = 0.279 #+ aaa * 0.069
Omega_b = 0.046 #+ aaa * 0.0042
sigma_8 = 0.82 #+ aaa * .0725
n_s = 0.97 #+ aaa * 0.14309020171
h_100 = 0.7 #+ aaa * 0.054

# Buzzard
Omega_m = 0.286  # + aaa * 0.069
Omega_b = 0.047  # + aaa * 0.0042
sigma_8 = 0.82  # + aaa * .0725
n_s = 0.96  # + aaa * 0.14309020171
h_100 = 0.7  # + aaa * 0.054


# bins to compute *****************
# IA params
A0 = 0.  # .#1.
z0 = 0.  # 62
alpha0 = 0.
z_shift = 0. * np.array([0.02, 0.02, 0.02, 0.02, 0.02]).astype(np.double)
z_spread = 1. * np.array([1., 1., 1., 1., 1.]).astype(np.double)

outp_folder = './Cl/'


name_file = '/global/project/projectdirs/des/mgatti/Moments_analysis/Nz/FLASK_'

red_bins = ['1', '2', '3', '4']

bins_array_true = [[3,3]]




bins_array_3 = []






for bb in bins_array_true:
    bins_array_2 = [bb]

    lensing_A = 1  # 1 lensed, 0 matter
    lensing_B = 1
    zsf = (z_shift).ctypes.data_as(POINTER(c_double))
    zsp = (z_spread).ctypes.data_as(POINTER(c_double))

    if not os.path.exists(outp_folder):
        os.mkdir(outp_folder)

    bins_array_2_o = np.array(bins_array_2)
    bins_array_3_o = np.array(bins_array_3)
    bins_array_2 = bins_array_2_o.flatten()
    bins_array_3 = bins_array_3_o.flatten()



    label_output = "CL_TOT_"

    pofz_file1 = []
    for bin in red_bins:
        pofz_file1.append(ctypes.c_char_p((name_file + bin + '.txt').encode('utf-8')))

    select_type = (c_char_p * len(pofz_file1))
    select = select_type()
    for key, item in enumerate(pofz_file1):
        select[key] = item

    print('Computing Cls ****** \n')
    # print('kappa\n')

    save_out_list2, select_type2 = make_output_list(bins_array_2_o, bins_array_3_o, lensing_A, lensing_B,label_output)
    print_Cls = lib.print_Cls
    print_Cls.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                          select_type,c_int,
                          POINTER(c_long), ctypes.c_int, POINTER(c_long), ctypes.c_int,
                          select_type2,ctypes.c_int, ctypes.c_int,
                          ctypes.c_double,ctypes.c_double,ctypes.c_double, POINTER(c_double), POINTER(c_double)]


    print_Cls(Omega_m,Omega_b,sigma_8,n_s,h_100,
    select, len(pofz_file1),
    bins_array_2.astype(np.long).ctypes.data_as(POINTER(c_long)),int(len(bins_array_2) / 2),  bins_array_3.astype(np.long).ctypes.data_as(POINTER(c_long)),int(len(bins_array_3) / 3),
    save_out_list2,lensing_A,lensing_B,
    A0, z0, alpha0, zsf, zsp)

    print_Cls.restype = ctypes.c_double


    
