import numpy as np
import os
import pdb
from cosmosis.datablock import names, option_section
import pickle
import numpy as np
import copy
import numpy as np
from scipy.interpolate import interp1d
import sys
import emcee
import h5py as h5
from Moments_analysis.compute_theory import *
from Moments_analysis.setup_runs_utilities import *
import pickle
import xarray as xr

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()



def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')

            
def covariance_jck(vec,jk_r,type_cov):

  #  Covariance estimation
    len_w = len(vec[:,0])
    jck = len(vec[0,:])
    
    err = np.zeros(len_w)
    mean = np.zeros(len_w)
    cov = np.zeros((len_w,len_w))
    corr = np.zeros((len_w,len_w))
    
    mean = np.zeros(len_w)
    for i in range(jk_r):
        mean += vec[:,i]
    mean = mean/jk_r
    
    if type_cov=='jackknife':
        fact=(jk_r-1.)/(jk_r)

    elif type_cov=='bootstrap':
        fact=1./(jk_r-1)
    
    for i in range(len_w):
        for j in range(len_w):
            for k in range(jck):
                cov[i,j] += (vec[i,k]- mean[i])*(vec[j,k]- mean[j])*fact

    for ii in range(len_w):
        err[ii]=np.sqrt(cov[ii,ii])

  #compute correlation
    for i in range(len_w):
        for j in range(len_w):
            corr[i,j]=cov[i,j]/(np.sqrt(cov[i,i]*cov[j,j]))


    return {'cov' : cov,
          'err' : err,
          'corr':corr,
          'mean':mean}



def make_covariance_xipxim(nrel, xipxim, pairs, scale_cut, theta, obs_2p):
    dvs = []
    # stack DV for covariance ********
    for j in range(nrel):
        # xip part
        for i, pair in enumerate(pairs):
            try:
                mask = (theta >scale_cut['angle_range_xip_{0}_{1}'.format(pair[1],pair[0])][0]) & (theta <scale_cut['angle_range_xip_{0}_{1}'.format(pair[1],pair[0])][1])
            except:
                mask = (theta >scale_cut['angle_range_xip_{1}_{0}'.format(pair[1],pair[0])][0]) & (theta <scale_cut['angle_range_xip_{1}_{0}'.format(pair[1],pair[0])][1])
            if i ==0:
                
                dv = xipxim[j]['{0}_{1}'.format(pair[0]-1,pair[1]-1)][0][mask]
            else:
                dv = np.hstack((dv,xipxim[j]['{0}_{1}'.format(pair[1]-1,pair[0]-1)][0][mask]))
        for i, pair in enumerate(pairs):
            try:
                mask = (theta >scale_cut['angle_range_xim_{0}_{1}'.format(pair[1],pair[0])][0]) & (theta <scale_cut['angle_range_xim_{0}_{1}'.format(pair[1],pair[0])][1])
            except:
                mask = (theta >scale_cut['angle_range_xim_{1}_{0}'.format(pair[1],pair[0])][0]) & (theta <scale_cut['angle_range_xim_{1}_{0}'.format(pair[1],pair[0])][1])
            dv = np.hstack((dv,xipxim[j]['{0}_{1}'.format(pair[1]-1,pair[0]-1)][1][mask]))
        dvs.append(dv)
    cc = covariance_jck(np.array(dvs).T,nrel,'bootstrap')
    
    # stack obs DV
    
    for i, pair in enumerate(pairs):
            mask_1 = (obs_2p[2].data['BIN1'] == pair[0]) & (obs_2p[2].data['BIN2'] == pair[1])
            try:
                mask = (theta >scale_cut['angle_range_xip_{0}_{1}'.format(pair[1],pair[0])][0]) & (theta <scale_cut['angle_range_xip_{0}_{1}'.format(pair[1],pair[0])][1])
            except:
                mask = (theta >scale_cut['angle_range_xip_{1}_{0}'.format(pair[1],pair[0])][0]) & (theta <scale_cut['angle_range_xip_{1}_{0}'.format(pair[1],pair[0])][1])
            if i ==0:
                mask_v = copy.copy(mask)
                dv_obs = mute[2].data['VALUE'][mask_1][mask]
            else:
                mask_v = np.hstack((mask_v,copy.copy(mask)))
                dv_obs = np.hstack((dv,mute[2].data['VALUE'][mask_1][mask]))
    for i, pair in enumerate(pairs):
            mask_1 = (obs_2p[3].data['BIN1'] == pair[0]) & (obs_2p[3].data['BIN2'] == pair[1])
            
            try:
                mask = (theta >scale_cut['angle_range_xim_{0}_{1}'.format(pair[1],pair[0])][0]) & (theta <scale_cut['angle_range_xim_{0}_{1}'.format(pair[1],pair[0])][1])
            except:
                mask = (theta >scale_cut['angle_range_xim_{1}_{0}'.format(pair[1],pair[0])][0]) & (theta <scale_cut['angle_range_xim_{1}_{0}'.format(pair[1],pair[0])][1])
            dv_obs = np.hstack((dv,mute[3].data['VALUE'][mask_1][mask]))
            mask_v = np.hstack((mask_v,copy.copy(mask)))
            
    cov_theo = obs_2p[1].data[:400,:][:,:400]
    cov_theo = cov_theo[mask_v,:][:,mask_v]
    return cc, np.array(dvs).T, dv_obs,cov_theo


    
scale_cut = dict()
scale_cut['angle_range_xip_1_1'] = [2.475 ,999.0]
scale_cut['angle_range_xip_1_2'] = [6.21691892, 999.0]
scale_cut['angle_range_xip_1_3'] = [6.21691892, 999.0]
scale_cut['angle_range_xip_1_4'] = [4.93827423, 999.0]
scale_cut['angle_range_xip_2_2'] = [6.21691892, 999.0]
scale_cut['angle_range_xip_2_3'] = [6.21691892, 999.0]
scale_cut['angle_range_xip_2_4'] = [6.21691892, 999.0]
scale_cut['angle_range_xip_3_3'] = [6.21691892, 999.0]
scale_cut['angle_range_xip_3_4'] = [6.21691892, 999.0]
scale_cut['angle_range_xip_4_4'] = [4.93827423, 999.0]
scale_cut['angle_range_xim_1_1'] = [24.75 ,999.0]
scale_cut['angle_range_xim_1_2'] = [62.16918918, 999.0]
scale_cut['angle_range_xim_1_3'] = [62.16918918, 999.0]
scale_cut['angle_range_xim_1_4'] = [49.3827423 ,999.0]
scale_cut['angle_range_xim_2_2'] = [62.16918918, 999.0]
scale_cut['angle_range_xim_2_3'] = [78.26637209, 999.0]
scale_cut['angle_range_xim_2_4'] = [78.26637209, 999.0]
scale_cut['angle_range_xim_3_3'] = [78.26637209, 999.0]
scale_cut['angle_range_xim_3_4'] = [78.26637209, 999.0]
scale_cut['angle_range_xim_4_4'] = [62.16918918, 999.0]

    
def setup(options):

    
    path_flask_xipxim =  options.get_string(option_section, "path_flask_xipxim", default = "/global/cscratch1/sd/mgatti/Mass_Mapping/moments/flask_tests/")
    probe1 =  options.get_string(option_section, "probe1", default = "2")
    key1 =  options.get_string(option_section, "key1", default = "2UNBlinded_24")
    probe2 =  options.get_string(option_section, "probe2", default = "3")
    key2 =  options.get_string(option_section, "key2", default = "3UNBlinded_24")
    nrel =  options.get_int(option_section, "nrel", default = 400)
    ndraws =  options.get_int(option_section, "ndraws", default = 1000)
    output_info =  options.get_string(option_section, "output_info", default = "info_dummy")
    
    path_chains =  options.get_string(option_section, "path_chains", default = "3UNBlinded_24")
    
    moments_conf = load_obj(path_chains)
    # load xipxim cov ***********
    xipxim = []
    for i in range(nrel):
        try:
            mute = load_obj(output+'/results_xipxim_{0}'+str(i+1))    
            xipxim.append(mute)
        except:
            pass
        
    probes =[probe1,probe2]
    keys =[key1,key2]
    info_PPD = dict()
    for i, probe in enumerate(probes):
        
        if probe == 'xipxim':
            # read file
            mute = pf.open(keys[i])
            b1 = mute[2].data['BIN1']
            b2 = mute[2].data['BIN2']
            pairs = np.array(([[b1[u],b2[u]] for u in range(len(b1))]))
            pairs = np.unique(pairs,axis=0)
            theta = mute[2].data['ANG'][:20]
            cc, info_PPD['xj_{0}'.format(i+1)], info_PPD['yobs_{0}'.format(i+1)],cov_theo = make_covariance_xipxim(nrel, xipxim, pairs, scale_cut, theta, mute)
            info_PPD['cov_{0}'.format(i+1)] = cc['cov']
            info_PPD['cov_t_{0}'.format(i+1)] = cov_theo
        elif probe == 'second':
            info = moments_conf[keys[i]]
            xj = info['dv_j_cov']
            info_PPD['cov_{0}'.format(i+1)] = info['cov']
            info_PPD['yobs_{0}'.format(i+1)] = info["y_obs"]
            info_PPD['xj_{0}'.format(i+1)] = np.matmul(info["transf_matrix"],info['dv_j_cov'].T)
            info_PPD['type_{0}'.format(i+1)] ='moments'
            info_PPD['config_{0}'.format(i+1)] = {'bins': info['bins'],'scales':info['scales'],'emu_setup':info['emu_config'],'tm':info['transf_matrix']}                  
        elif probe == 'second_third':
            info = moments_conf[keys[i]]
            xj = info['dv_j_cov']
            info_PPD['cov_{0}'.format(i+1)] = info['cov']
            info_PPD['yobs_{0}'.format(i+1)] = info["y_obs"]
            info_PPD['xj_{0}'.format(i+1)] = np.matmul(info["transf_matrix"],info['dv_j_cov'].T)
            info_PPD['type_{0}'.format(i+1)] ='moments'
            info_PPD['config_{0}'.format(i+1)] = {'bins': info['bins'],'scales':info['scales'],'emu_setup':info['emu_config'],'tm':info['transf_matrix']}            
        elif probe == 'third':
            info = moments_conf[keys[i]]
            xj = info['dv_j_cov']
            info_PPD['cov_{0}'.format(i+1)] = info['cov']
            info_PPD['yobs_{0}'.format(i+1)] = info["y_obs"]
            info_PPD['xj_{0}'.format(i+1)] = np.matmul(info["transf_matrix"],info['dv_j_cov'].T)
            info_PPD['type_{0}'.format(i+1)] ='moments'
            info_PPD['config_{0}'.format(i+1)] = {'bins': info['bins'],'scales':info['scales'],'emu_setup':info['emu_config'],'tm':info['transf_matrix']}            
    
        
    #make joint covariance
    uu = np.vstack([info_PPD['xj_1'][:,:nrel],info_PPD['xj_2'][:,:nrel]])
    cc = covariance_jck(uu,nrel,'bootstrap')
    info_PPD['cov12'] = cc['cov']
    # update with theory cov if possible
    try:
        sh = info_PPD['xj_1'].shape[0]
        info_PPD['cov12'][:sh,:][:,:sh] = info_PPD['cov_t_1'.format(i+1)]
    except:
        pass
    try:
        sh = info_PPD['xj_1'].shape[0]
        sh2 = info_PPD['xj_2'].shape[0]
        info_PPD['cov12'][sh:sh+sh2,:][:,sh:sh+sh2] = info_PPD['cov_t_2'.format(i+1)]
    except:
        pass
    info_PPD['ndraws'] = ndraws
    info_PPD['cov12_cross'] = info_PPD['cov12'][:sh,:][:,sh:sh+sh2].T
    info_PPD['output_info'] = output_info
    return info_PPD

def compute_th_moments(block,config):
    
    params = dict()
    params['Omega_m'] = block['cosmological_parameters','omega_m']
    params['Sigma_8'] = block['cosmological_parameters','sigma_8']
    params['n_s'] = block['cosmological_parameters','n_s']
    params['Omega_b'] = block['cosmological_parameters','omega_b']
    params['h100'] = block['cosmological_parameters','h0']
    params['A_IA'] = block['intrinsic_alignment_parameters','A1']
    params['alpha0'] = block['intrinsic_alignment_parameters','alpha1']

    # read Nz and interpolate it ****************************************************************


    nbin = block['nz_source', "nbin"]
    z = block['nz_source', "z"]
    Nz0 = []

    for i in range(1, nbin + 1):
        nz = block['nz_source', "bin_%d" % i]
        fz = interp1d(z,nz)
        Nz0.append(fz)
    config["Nz0"] = Nz0
    
    # m and dz must be re-collected into 1 array.
    m = []
    dz = []
    for i in range(len(Nz0)):
        m.append(block['shear_calibration_parameters','m{0}'.format(i+1)])
        dz.append(0.)
    params['m'] = m
    params['dz'] = dz


    y,_ = compute_theory(config['Nz0'],config['bins'],config['scales'],params,config['emu_setup'])
    y = np.matmul(config['tm'],y)
    return y
    

def execute(block, config):
    
    # *******************************************************
    # Read covariances

    cov_d = config['cov_1']
    inv_cov_d = np.linalg.inv(cov_d)
    theory_d = compute_th_moments(block,config['config_1'])
    obs_d = config['yobs_1']
    

    chi2_realizations = []
    for k in range(100):
        d_realization = np.random.multivariate_normal(theory_d, cov_d)
        diff_data = obs_d - theory_d
        chi2_data = np.dot(diff_data, np.dot(inv_cov_d, diff_data))
        diff_realization = d_realization - theory_d
        chi2_realization = np.dot(diff_realization, np.dot(inv_cov_d, diff_realization))
        chi2_realizations.append(chi2_realization)
    chi2_realizations = np.array(chi2_realizations)

    print ('boiaAA',len(chi2_realizations[chi2_realizations>chi2_data]),len(chi2_realizations))
     
    info_to_save = dict()
    info_to_save['p_v'] = len(chi2_realizations[chi2_realizations>chi2_data])*1./len(chi2_realizations)

    save_obj('./dummy/{0}'.format(config['output_info']),info_to_save)
    # un-compressed DVs
    

    
    
 
    return 0