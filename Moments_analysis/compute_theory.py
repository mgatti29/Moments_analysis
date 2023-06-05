import time
import numpy as np
import os
from scipy.interpolate import interp1d
import timeit
import numpy.random as rng
import scipy.stats as stats
from glob import glob
import timeit
import numpy as np
from multiprocessing import Pool,sharedctypes
from functools import partial
from contextlib import closing
import math
import shutil
import pandas as pd
import ctypes
from ctypes import *
import h5py
import cosmolopy.distance as cd
import cosmolopy
import scipy
from scipy.interpolate import interp1d
import timeit
import pickle


'''
Routines to save / load pickle.


'''
def save_obj(name, obj):
    """
    Save an object to a pickle file.

    Args:
        name (str): Name of the file.
        obj (object): Object to be saved.
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def load_obj(name):
    """
    Load an object from a pickle file.

    Args:
        name (str): Name of the file.

    Returns:
        object: Loaded object.
    """
    try:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)  # , encoding='latin1')
    except:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f, encoding='latin1')

                
def compute_theory(Nz0, bins_dict, scales,params,emu_dict):
    
    '''
    It computes theory predictions given a N(z), parameters, bins, smoothing scales and emulator dictionary.
    
    Example of usage:
    
    params = dict()
    params['Omega_m'] = 0.28
    params['Sigma_8'] = 0.82
    params['h100'] = 0.7 
    params['Omega_b'] = 0.047
    params['n_s'] = 0.97
    params['dz'] = np.array([0.0,0.0,0.,0.])
    params['m'] = np.array([-1.,-1.6,-2.5,-3.8])*0.01


    path_redshift = "/global/project/projectdirs/des/mgatti/Moments_analysis/Nz/DESY3_unblind"
    Nz0 = load_Nz(['1','2','3','4'],path_redshift)
    
    bins_dict = dict()
    bins_dict['2 + 3 tomo cross'] = {'bins':[[1,1],[2,2],[3,3],[4,4]]} 
    
    scales = np.array([3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.])
    
    z_EMU = np.loadtxt("/global/project/projectdirs/des/mgatti/Moments_analysis/z_EMU.txt")
    emu_dict=dict()
    emu_dict['svd_train'] =svd_train
    emu_dict['z_EMU'] = z_EMU
    
    vector, m_dict = compute_theory(Nz0, bins_dict, scales, params,emu_dict)
    '''
    
    def lensing_kernel23_fast(dave_lr,dchislr,dave_hr,dchishr,zhr,Nz):
        '''
        This computes the lensing kernel. 
        It actually calls a cython gunction that does the double summation.
        '''
        array_type_lr = ctypes.c_double * len(dave_lr)
        array_type_hr = ctypes.c_double * len(dave_hr)

        dlr = array_type_lr(0.)
        dclr = array_type_lr(0.)

        for i in range(len(dave_lr)):
                dlr[i] = dave_lr[i]
                dclr[i] = dchislr[i]



        dhr = array_type_hr(0.)
        dchr = array_type_hr(0.)
        zzhr= array_type_hr(0.)
        for i in range(len(dave_hr)):
                dhr[i] = dave_hr[i]
                dchr[i] = dchishr[i]
                zzhr[i] = zhr[i]


        dnmute = array_type_lr(0.)
        normd = np.sum(nzm_m)
        nzm_mm = Nz/normd/dchislr
        for jjj in range(len(dave_lr)):
            dnmute[jjj] = nzm_mm[jjj]


        print_lk = lib2.lak
        print_lk.argtypes = [ ctypes.c_int,   ctypes.c_int, array_type_lr ,array_type_lr ,array_type_hr,array_type_hr,array_type_hr,array_type_lr,array_type_hr ]
        print_lk.restype = ctypes.c_double
        weigh = (c_double*len(dave_hr))()

        print_lk(len(dlr),len(dhr),dlr,dclr,dhr ,dchr ,zzhr,dnmute,weigh)
        
        '''
        This is what the cython script does in practice.
        weigh1 = np.zeros(len(dhr))
        for jj in range(1,len(dlr)):
            for ii in range(1,len(dhr)):
                if (dave_lr[jj]>=dave_hr[ii]):
                    weigh1[ii] += dchislr[jj]*nzm_mm[jj]*(dave_lr[jj]-dave_hr[ii])/(dave_lr[jj])*(1.+zhr[ii])#*dchishr[ii]


        
        '''
        return weigh
                
    # check if all parameters are here ***************
    if not ('Omega_m' in params.keys()):
        params["Omega_m"] = 0.3
    if not ('Sigma_8' in params.keys()):
        params["Sigma_8"] = 0.8
    if not ('n_s' in params.keys()):
        params["n_s"] = 0.97
    if not ('Omega_b' in params.keys()):
        params["Omega_b"] = 0.046
    if not ('h100' in params.keys()):
        params["h100"] = 0.7
    if not ('dz' in params.keys()):
        params["dz"] = np.zeros(len(Nz0))
    if not ('m' in params.keys()):
        params["m"] = np.zeros(len(Nz0))
    if not ('A_IA' in params.keys()):
        params["A_IA"] = 0.
    if not ('alpha0' in params.keys()):
        params["alpha0"] = 0.
    #if not ('z0' in params.keys()):
    #    params["z0"] = 0.67
    z0 = 0.67
       
    # load double cycle ***************************
    lib2=ctypes.cdll.LoadLibrary("./double_cycle.so")
    
    # cosmo  this is needed to compute distances.
    cosmo = {'omega_M_0':params["Omega_m"] , 
                     'omega_lambda_0':1-params["Omega_m"] ,
                     'omega_k_0': 0.0, 
                     'omega_b_0' : params["Omega_b"] ,
                     'h':params["h100"],
                     'sigma_8' : params["Sigma_8"],
                     'n': params["n_s"]}
    

    # cosmo for scale cut (at fixed cosmology!):
    cosmo_scale = {'omega_M_0':0.28 , 
                     'omega_lambda_0':1 - 0.28 ,
                     'omega_k_0': 0.0, 
                     'omega_b_0' : 0.047 ,
                     'h':0.7 ,
                     'sigma_8' :  0.82,
                     'n': 0.97}
    
    z = emu_dict['z_EMU']
    svd_train =  emu_dict['svd_train'] 
                
    growth = (cosmolopy.perturbation.fgrowth(z,cosmo['omega_M_0']))
    dchis = 1./cd.e_z(z,**cosmo)#/(100*cosmo['h'])
    dchism = 1./cd.e_z(z+0.5*(z[1]-z[0]),**cosmo)#/(100*cosmo['h'])
    
    # interpolation of the distance **********
    nhr = 5000
    zhr = np.arange(nhr)*1.2*z[-1]/(nhr*1.)

    dddd = 1./cd.e_z(zhr,**cosmo)*(zhr[1]-zhr[0])#*(299792.458/(100*cosmo['h']))
    xxxx =  scipy.integrate.cumtrapz(dddd)
    interp_dist = interp1d(zhr[1:],xxxx)


    dist= interp_dist(z)
    d_ave= interp_dist(z+0.5*(z[1]-z[0]))



    nhr = 2000
    zhr = np.arange(nhr)*1.*z[-1]/(nhr*1.)
    dave_hr =  interp_dist(zhr+0.5*(zhr[1]-zhr[0]))
    dchishr = 1./cd.e_z(zhr,**cosmo)


    nlr = 500 # to better res, around 500
    zlr = np.arange(nlr)*1.*z[-1]/nlr
    dave_lr = interp_dist(zlr+0.5*(zlr[1]-zlr[0]))
    dchislr = 1./cd.e_z(zlr,**cosmo)

    
    

    
    # load nz ***************************************
    Nz = []
    Qz = []
    for i,dz0 in enumerate(params['dz']):
        if dz0>0.2:
            dz0=0.2
        if dz0<-0.2:
            dz0=-0.2
        mask = (z+dz0>=0.)&(z+dz0<=2.45)
        nzm = Nz0[i]((z+dz0)[mask])
        normd = np.sum(nzm)*(z[1]-z[0])
        nzm_m = np.zeros(len(z))
        nzm_m [mask]=nzm
        Nz.append([nzm_m/normd,mask])

        masku = (zlr+dz0>=0.)&(zlr+dz0<=2.45)
        nzm = Nz0[i]((zlr+dz0)[masku])
        nzm_m = np.zeros(len(zlr))
        nzm_m[masku]=nzm


        qmm1a = params["Omega_m"]*np.array(list(lensing_kernel23_fast(dave_lr,dchislr,dave_hr,dchishr,zhr,nzm_m)))

        #print qmm1a,qmm1ab
        fq = interp1d(zhr, qmm1a*1.5)    
        qmm1=  fq(z[1:-1])
        qmm = np.zeros(len(qmm1)+2)
        qmm[0],qmm[1:-1]=qmm1a[0]*1.5,qmm1[:]


        Qz.append(np.array(qmm))



    # EMULATOR ******************************************************
    input_cosmology =[np.array([params["Sigma_8"],params["Omega_m"],params["Omega_b"],params["n_s"],params["h100"]])]
    smoothing_scales_emu = np.array([3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.])

    d2_array = []
    d3_array = []
    d3_array_c = []

    for sm in np.arange(len(smoothing_scales_emu))[np.in1d(smoothing_scales_emu,scales)]:
       
        dim = len(svd_train[sm][0][-1])
        v0 = []
        for i in range(dim):
            YY_std = svd_train[sm][0][4][i]
            YY_mean = svd_train[sm][0][3][i]
            YY_in = (svd_train[sm][0][2][i]-YY_mean)/YY_std
            v0.append((svd_train[sm][0][-1][i].predict(YY_in, input_cosmology)[0])*YY_std+YY_mean)
        v0 = np.array(v0).reshape(dim)
        d2_array.append(np.exp(np.array(np.dot(svd_train[sm][0][0][:,:dim] * svd_train[sm][0][1][:dim], v0))))
    d2 = np.array(d2_array).T

    for sm in np.arange(len(smoothing_scales_emu))[np.in1d(smoothing_scales_emu,scales)]:
        dim = len(svd_train[sm][2][-1])
        v0 = []
        for i in range(dim):
            YY_std = svd_train[sm][2][4][i]
            YY_mean = svd_train[sm][2][3][i]
            YY_in = (svd_train[sm][2][2][i]-YY_mean)/YY_std
            v0.append((svd_train[sm][2][-1][i].predict(YY_in, input_cosmology)[0])*YY_std+YY_mean)
        v0 = np.array(v0).reshape(dim)
        d3_array.append(np.exp(np.dot(svd_train[sm][2][0][:,:dim] * svd_train[sm][2][1][:dim], v0)))

    try:
        for sm in np.arange(len(smoothing_scales_emu))[np.in1d(smoothing_scales_emu,scales)]:
            dim = len(svd_train[sm][4][-1])
            v0 = []
            for i in range(dim):
                YY_std = svd_train[sm][4][4][i]
                YY_mean = svd_train[sm][4][3][i]
                YY_in = (svd_train[sm][4][2][i]-YY_mean)/YY_std
                v0.append((svd_train[sm][4][-1][i].predict(YY_in, input_cosmology)[0])*YY_std+YY_mean)
            v0 = np.array(v0).reshape(dim)
            d3_array_c.append(np.exp(np.dot(svd_train[sm][4][0][:,:dim] * svd_train[sm][4][1][:dim], v0)))
        d3 = (np.array(d3_array)*np.array(d3_array_c)).T
    except:
        d3 = (np.array(d3_array)).T


    

    count = 0 
    m_dict = dict()
    for i,binx in enumerate(bins_dict['bins']):
        if len(binx) == 2:
            IA = params["A_IA"]*(((1+z[1:])/(1+z0))**params["alpha0"])*0.0134*params["Omega_m"]/growth[1:]
            IA1 = params["A_IA"]*(((1+z[1:])/(1+z0))**params["alpha0"])*0.0134*params["Omega_m"]/growth[1:]
    
            weight1 = (1.+params["m"][binx[0]-1])*(Qz[binx[0]-1][1:]*dist[1:]- Nz[binx[0]-1][0][1:]/(dchis[1:])*IA)
            weight2 = (1.+params["m"][binx[1]-1])*(Qz[binx[1]-1][1:]*dist[1:]- Nz[binx[1]-1][0][1:]/(dchis[1:])*IA1)
            mute = weight1*weight2
            zzm = 0.
            if bins_dict['scale_cut'] != None:
                for bx in binx:
                    zzm += bins_dict['Nz_mean'][bx-1]
                min_theta_rp=(bins_dict['scale_cut']/((1.+zzm/len(binx))*cd.comoving_distance(zzm/len(binx),**cosmo_scale)*(2*math.pi)/360)*60)
                mask_scales =np.array(scales)>min_theta_rp   
            else:
                mask_scales =  np.array(scales) == np.array(scales)
            
            dd2 = 2*np.dot((d2[1:,:]).T,mute*dchis[1:]*(z[1]-z[0]))
            m_dict['{0}_{1}'.format(binx[0],binx[1])] = dd2
            m_dict['{1}_{0}'.format(binx[0],binx[1])] = dd2
            if count == 0:
                vector = dd2[mask_scales]
            else:
                vector = np.hstack([vector,dd2[mask_scales]])
            count +=1
        if len(binx) == 3:
           
            IA =  params["A_IA"]*(((1+z[1:])/(1+z0))**params["alpha0"])*0.0134*params["Omega_m"]/growth[1:]
            IA1 = params["A_IA"]*(((1+z[1:])/(1+z0))**params["alpha0"])*0.0134*params["Omega_m"]/growth[1:]
            IA2 = params["A_IA"]*(((1+z[1:])/(1+z0))**params["alpha0"])*0.0134*params["Omega_m"]/growth[1:]


            weight1 = (1.+params["m"][binx[0]-1])*(Qz[binx[0]-1][1:]*dist[1:] - Nz[binx[0]-1][0][1:]/(dchis[1:])*IA )
            weight2 = (1.+params["m"][binx[1]-1])*(Qz[binx[1]-1][1:]*dist[1:] - Nz[binx[1]-1][0][1:]/(dchis[1:])*IA1 )
            weight3 = (1.+params["m"][binx[2]-1])*(Qz[binx[2]-1][1:]*dist[1:] - Nz[binx[2]-1][0][1:]/(dchis[1:])*IA2 )
            mute = weight1*weight2*weight3
            dd3 = 6*np.dot((d3[1:,:]).T,mute*dchis[1:]*(z[1]-z[0]))
            zzm = 0.
            if bins_dict['scale_cut'] != None:
                for bx in binx:
                    zzm += bins_dict['Nz_mean'][bx-1]
                min_theta_rp=(bins_dict['scale_cut']/((1.+zzm/len(binx))*cd.comoving_distance(zzm/len(binx),**cosmo_scale)*(2*math.pi)/360)*60)
                mask_scales =np.array(scales)>min_theta_rp   
            else:
                mask_scales =  np.array(scales) == np.array(scales)
            
            if count == 0:
                vector = dd3[mask_scales]
            else:
                vector = np.hstack([vector,dd3[mask_scales]])
                
            # all the permutations : 
            m_dict['{0}_{1}_{2}'.format(binx[0],binx[1],binx[2])] = dd3
            m_dict['{0}_{2}_{1}'.format(binx[0],binx[1],binx[2])] = dd3
            m_dict['{1}_{0}_{2}'.format(binx[0],binx[1],binx[2])] = dd3
            m_dict['{1}_{2}_{0}'.format(binx[0],binx[1],binx[2])] = dd3
            m_dict['{2}_{0}_{1}'.format(binx[0],binx[1],binx[2])] = dd3
            m_dict['{2}_{1}_{0}'.format(binx[0],binx[1],binx[2])] = dd3

            count +=1

    return vector,m_dict 
            
            
  
def load_Nz(bins_theory, path_redshift):
    """
    Load the Nz (redshift distribution) data from files for different bins.

    Parameters:
        bins_theory (list): List of bin indices.
        path_redshift (str): Path to the redshift file.

    Returns:
        list: List of interpolation functions for Nz.

    """
    # Load Nz ********
    Nz0 = []
    for i, binx in enumerate(bins_theory):
        # Construct the path to the Nz file
        path = '{1}_{0}.txt'.format(binx, path_redshift)
        
        # Load data from file
        mute = np.loadtxt(path)
        
        # Extract redshift and corresponding values
        z_m, f_m = np.zeros(len(mute[:, 0]) + 1), np.zeros(len(mute[:, 0]) + 1)
        z_m[1:] = mute[:, 0]
        f_m[1:] = mute[:, 1]

        # Interpolate the redshift distribution
        fz = interp1d(z_m, f_m)
        Nz0.append(fz)
    
    return Nz0
    

 
'''
The following routines are a python version of the second and third moments computation implemented in the c++ code. Not fully tested (second moments are tested, 3rd moments not fully.)
'''

import scipy
def compute_masked_2m(P_lz, smoothing_scales, lmax=2048):
    """
    Compute the smoothed (by a top-hat filter) 2nd moments of the density field given the
    3D power spectrum at fixed z. (l=k/chi(z)).

    Args:
        P_lz (array-like): 3D power spectrum at fixed z.
        smoothing_scales (array-like): Smoothing scales in arcminutes.
        lmax (int, optional): Maximum value of ell. Defaults to 2048.

    Returns:
        array: Array of computed moments for each smoothing scale.
    """
    moments = []
    ell = np.arange(lmax)
    for i, sm in enumerate(smoothing_scales):
        sm_rad = (sm / 60.) * np.pi / 180.

        # Smoothing kernel (top-hat)
        A = 1. / (2 * np.pi * (1. - np.cos(sm_rad)))
        B = np.sqrt(np.pi / (2. * ell + 1.0))
        fact = -B * (scipy.special.eval_legendre(ell + 1, np.cos(sm_rad)) - scipy.special.eval_legendre(ell - 1, np.cos(sm_rad))) * A
        fact[0] = 1

        moments.append((1.) * np.sum(fact[:lmax] ** 2 * P_lz[:lmax]))

    return np.array(moments)


def compute_masked_3m(P_lz, smoothing_scales, P_lz_masked=None, lmax=2048, sm_formulae='SC', z=0.5,
                      cosmo={'baryonic_effects': False, 'omega_n_0': 0., 'N_nu': 0, 'omega_M_0': 0.3,
                             'omega_b_0': 0.004, 'sigma_8': 0.8, 'omega_lambda_0': 1 - 0.3, 'omega_k_0': 0.0,
                             'h': 0.72, 'n': 0.92}, expansion=False):
    """
    Compute the smoothed (by a top-hat filter) 3rd moments of the density field given the 3D power spectrum at fixed z.
    It implements small-scales fitting formulae.

    Args:
        P_lz (array-like): 3D power spectrum at fixed z.
        smoothing_scales (array-like): Smoothing scales in arcminutes.
        P_lz_masked (array-like, optional): Masked 3D power spectrum at fixed z. Defaults to None.
        lmax (int, optional): Maximum value of ell. Defaults to 2048.
        sm_formulae (str, optional): Smoothing formula. Defaults to 'SC'.
        z (float, optional): Redshift. Defaults to 0.5.
        cosmo (dict, optional): Cosmological parameters. Defaults to {'baryonic_effects': False, 'omega_n_0': 0.,
                              'N_nu': 0, 'omega_M_0': 0.3, 'omega_b_0': 0.004, 'sigma_8': 0.8,
                              'omega_lambda_0': 1 - 0.3, 'omega_k_0': 0.0, 'h': 0.72, 'n': 0.92}.
        expansion (bool, optional): Expansion flag. Defaults to False.

    Returns:
        array: Array of computed moments for each smoothing scale.
    """
    d = cd.comoving_distance(z, **cosmo)
    ell = np.arange(lmax + 1)

    # This should be the NL power spectrum given as input ***
    Pl = cosmolopy.perturbation.power_spectrum(ell / d, z, **cosmo)
    # NL scale:
    l_nl = ell[(Pl * (ell / d) ** 3) / (2. * np.pi ** 2) > 1][0]

    ell = np.arange(lmax)
    # PS index.
    ns = np.diff((Pl)) / Pl[:lmax] * ell
    ns[ns != ns] = 1.

    # Growth
    Dp = cosmolopy.perturbation.fgrowth(z, cosmo['omega_M_0']) / (1. + z)

    # Initialise coefficients small-scales fitting formulae.
    if sm_formulae == 'SC':
        coeff = [0.25, 3.5, 2., 1., 2., -0.2, 1., 0., 0.]
    elif sm_formulae == 'GM':
        coeff = [0.484, 3.740, -0.849, 0.392, 1.013, -0.575, 0.128, -0.722, -0.926]
    if sm_formulae == 'NL':
        a = 1.
        b = 1.
        c = 1.
    else:
        # Transition l from linear to non-linear
        q = ell * 1. / l_nl

        a = (1. + ((cosmo['sigma_8'] * Dp) ** coeff[5]) * (0.7 * (4. - 2. ** ns) / (1. + 2. ** (2. * ns + 1))) ** 0.5 * (
                    q * coeff[0]) ** (ns + coeff[1])) / (1. + (q * coeff[0]) ** (ns + coeff[1]))
        b = (1. + 0.2 * coeff[2] * (ns + 3) * (q * coeff[6]) ** (ns + coeff[7] + 3)) / (
                    1. + (q * coeff[6]) ** (ns + coeff[7] + 3.5))
        c = (1. + 4.5 * coeff[3] / (1.5 + (ns + 3) ** 4) * (q * coeff[4]) ** (ns + 3 + coeff[8])) / (
                    1 + (q * coeff[4]) ** (ns + 3.5 + coeff[8]))
        a[0] = 1
        b[0] = 1
        c[0] = 1

    mu = 3. / 7.
    moments = []
    ell = np.arange(lmax)
    for i, sm in enumerate(smoothing_scales):
        sm_rad = (sm / 60.) * np.pi / 180.

        A = -1. / (2 * np.pi * (1. - np.cos(sm_rad)))
        B = np.sqrt(np.pi / (2. * ell + 1.0))
        B1 = np.sqrt(np.pi * (2. * ell + 1.0))
        fact = B * (scipy.special.eval_legendre(ell + 1, np.cos(sm_rad)) - scipy.special.eval_legendre(ell - 1, np.cos(sm_rad))) * A
        d_fact = -B1 * A * np.sin(sm_rad) * scipy.special.eval_legendre(ell, np.cos(sm_rad))
        d_fact += A * np.pi * 2 * np.sin(sm_rad) * fact

        moment_2d = np.sum(fact[:lmax] ** 2 * P_lz[:lmax])
        moment_2d_a = np.sum(a * fact[:lmax] ** 2 * P_lz[:lmax])
        moment_2d_b = np.sum(b * fact[:lmax] ** 2 * P_lz[:lmax])
        moment_2d_c = np.sum(c * fact[:lmax] ** 2 * P_lz[:lmax])

        try:
            moment_2d_masked = np.sum(fact[:lmax] ** 2 * P_lz_masked[:lmax])

        except:
            moment_2d_masked = np.sum(fact[:lmax] ** 2 * P_lz[:lmax])

        dvL_dlnR = sm_rad * np.sum(b * fact * d_fact * P_lz[:lmax])
        dlnvL_dlnR = dvL_dlnR / moment_2d_b

        m3 = 6 * moment_2d_masked ** 2 * (
                    0.5 * (2 * mu * moment_2d_a ** 2 + (1. - mu) * moment_2d_c ** 2) + moment_2d_b ** 2 * 0.25 * dlnvL_dlnR) / (
                     moment_2d ** 2)
        if not expansion:
            moments.append(m3)
        else:
            # https://arxiv.org/pdf/astro-ph/9903486.pdf
            # There's one integral that in my paper which is actually approximated. /int dphi W const.. Here's the exact solution
            # fact_sp = copy.copy(fact[:lmax])

            moment2_2d_a_sp = 0.
            moment2_2d_b_sp = 0.
            moment2_2d_c_sp = 0.

            # First 2*3+1 Bessel functions. (only the first two matters..)
            for ii in range(0, 3):
                fact_sp = ((2. * ell + 1) * scipy.special.jv(1, sm_rad * ell) * scipy.special.jv(2 * ii + 1,
                                                                                                sm_rad * ell) / (
                                   sm_rad * ell) ** 2 / (np.pi))
                if ii == 0:
                    fact_sp = copy.copy(fact ** 2)
                else:
                    fact_sp[0] = 0.
                moment2_2d_a_sp += (2 * ii + 1) * np.sum(a * fact_sp[:lmax] * P_lz[:lmax]) ** 2
                moment2_2d_b_sp += (2 * ii + 1) * np.sum(b * fact_sp[:lmax] * P_lz[:lmax]) ** 2
                moment2_2d_c_sp += (2 * ii + 1) * np.sum(c * fact_sp[:lmax] * P_lz[:lmax]) ** 2

            m3 = moment_2d_masked ** 2 * 3 * (
                        2 * moment_2d_b ** 2 - (1. - mu) * moment_2d_c ** 2 + 2 * mu * moment2_2d_a_sp -
                        2 * moment2_2d_b_sp +
                        2 * (1. - mu) * moment2_2d_c_sp + moment_2d_b ** 2 * 0.5 * dlnvL_dlnR) / (moment_2d ** 2)
            moments.append(m3)

    return np.array(moments)

   
