from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d
import sys
import emcee
import h5py as h5
from Moments_analysis.compute_theory import *
from Moments_analysis.setup_runs_utilities import *
import pickle
import xarray as xr

import pickle

def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')
                

def setup(options):

    path_chains =  options.get_string(option_section, "path_chains", default = "")
    key =  options.get_string(option_section, "key", default = "")

    moments_conf = load_obj(path_chains)
    info = moments_conf[key]
    
    config = dict()
    config["Sellentin"] = info["Sellentin"]
 
    xx,inv_cov,scales,bins,emu_setup,y_obs,tm,dc= info['Nz'],info['inv_cov'],info['scales'],info['bins'],info['emu_config'],info['y_obs'],info["transf_matrix"],info["dc"]
    config["xx"] =  xx
    config["inv_cov"] =  inv_cov
    config["scales"] =  scales
    config["bins"] =  bins
    config["emu_setup"] =  emu_setup
    config["y_obs"] =  y_obs
    config["tm"] =  tm
    config["dc"] =  dc
    config["cov"] =  info['cov']
    
            
    return config

def execute(block, config):
    name_likelihood = 'moments_like'

    
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

    if config["dc"]:
        y = np.matmul(config['tm'],y)
    w = y-config['y_obs']

    chi2 = np.matmul(w,np.matmul(config['inv_cov'],w))


    if config['Sellentin']["S"]:
        like = -np.log(1.0 +chi2/(float(config['Sellentin']["Ns"]) - 1))*float(config['Sellentin']["Ns"])/2.0
        block[names.likelihoods, name_likelihood] = like
    else:
        block[names.likelihoods, name_likelihood] =  -0.5 * chi2- 0.5*np.log(np.linalg.det(config['cov']))

    return 0

def cleanup(config):
    pass
