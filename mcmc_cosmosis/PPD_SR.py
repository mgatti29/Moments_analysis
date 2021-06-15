
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

from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d


def get_ratio_from_gammat(gammat1, gammat2, inv_cov):
    #Given two gammats, calculate the ratio
    s2 = (1./float(np.matrix(np.ones(len(gammat1)))*np.matrix(inv_cov)*np.matrix(np.ones(len(gammat1))).T))
    ratio = s2*float(np.matrix(gammat1/gammat2)*np.matrix(inv_cov)*np.matrix(np.ones(len(gammat1))).T)
    return ratio

def setup(options):
    name_likelihood = options.get_string(option_section, "name_likelihood", default = "ratio_like")
    ratio_save_name = options.get_string(option_section, "save_ratio_filename", default = "None")
    gglensing_section = options.get_string(option_section, "gglensing_section", default="galaxy_shear_xi")
    ratio_measure_file = options.get_string(option_section, "measured_ratio_filename", default = "None")
    th_limit_low_s14 = options.get_double_array_1d(option_section, "th_limit_low_s14")
    th_limit_low_s24 = options.get_double_array_1d(option_section, "th_limit_low_s24")
    th_limit_low_s34 = options.get_double_array_1d(option_section, "th_limit_low_s34")
    th_limit_high = options.get_double_array_1d(option_section, "th_limit_high")
    lens_bins = options.get_int(option_section, "lens_bins")
    output_info =  options.get_string(option_section, "output_info", default = "info_dummy")
    
    #Load measured ratios and covariance
    if (not ratio_measure_file.lower() == "none"):
        ratio_data  = np.load(ratio_measure_file,allow_pickle=True, encoding='latin1').item()
        measured_ratios = ratio_data['measured_ratios']
        ratio_cov = ratio_data['ratio_cov']
        theta_data = ratio_data['theta_data']
        ratio_invcov = np.linalg.inv(ratio_cov)
        inv_cov_individual_ratios = ratio_data['inv_cov_individual_ratios']
        logdetC = np.log(np.linalg.det(ratio_cov))
    else:
        #Make up ratio values for testing
        measured_ratios = 1.
        ratio_invcov = 1.

    # small scales ratios have different scale cuts depending on the sources, while large scales scale cuts only depend on the lens
    th_limit_low = [th_limit_low_s14, th_limit_low_s24, th_limit_low_s34]

    config_data = {"name_likelihood":name_likelihood, "measured_ratios":measured_ratios, "theta_data":theta_data, "ratio_invcov":ratio_invcov, "ratio_save_name":ratio_save_name, "gglensing_section":gglensing_section, "th_limit_low":th_limit_low, "th_limit_high":th_limit_high, "lens_bins":lens_bins, "inv_cov_individual_ratios":inv_cov_individual_ratios, "logdetC":logdetC,'output_info':output_info}
    return config_data


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

    
    

def execute(block, config):
    name_likelihood = config['name_likelihood']
    measured_ratios =  config['measured_ratios']
    theta_data = config['theta_data']
    ratio_invcov  = config['ratio_invcov']
    gglensing_section = config['gglensing_section']
    th_limit_low = config['th_limit_low']
    th_limit_high = config['th_limit_high']
    lens_bins = config['lens_bins']
    inv_cov_individual_ratios = config['inv_cov_individual_ratios']
    logdetC = config['logdetC']

    #Compute ratio theory using gammat theory from block
    #nlens_bins = block[gglensing_section, 'nbin_a']
    nlens_bins = lens_bins
    nsource_bins = block[gglensing_section, 'nbin_b']
    bin_avg = block.get_bool(gglensing_section, 'bin_avg')#, default=False)
    #print('bin avg', bin_avg)

    s_ref = nsource_bins
    theta = block[(u'galaxy_shear_xi', u'theta')]*3437.75 #in arcmins
    #print("theta sr", theta)
    theory_ratios = []
    count = 0
    for li in range(1,nlens_bins+1):
        gammat_ref = block[gglensing_section, 'bin_' + str(li) + '_' + str(s_ref)]
        if not bin_avg:
            gammat_ref = interp1d(theta, gammat_ref)(theta_data) 
            print('interpolating, bin avg', bin_avg)

        for si in range(1,nsource_bins):
            maski = ((theta_data>th_limit_low[si-1][li-1])&(theta_data<=th_limit_high[li-1])&(gammat_ref!=0))

            inv_cov_ind_ratio = inv_cov_individual_ratios[count]
            inv_cov_ind_ratio = inv_cov_ind_ratio[maski].T[maski]
            gammati = block[gglensing_section, 'bin_' + str(li) + '_' + str(si)]
            if not bin_avg:
                gammati = interp1d(theta, gammati)(theta_data) 
                print('interpolating, bin avg', bin_avg)
            ratio = get_ratio_from_gammat(gammati[maski], gammat_ref[maski],inv_cov_ind_ratio)
            theory_ratios.append(ratio)
            count += 1
    theory_ratios = np.array(theory_ratios)

    #compute likelihood
    diff = measured_ratios - theory_ratios
    #print measured_ratios, theory_ratios, diff
    
    print (measured_ratios)
    print (theory_ratios)
    print (np.sqrt(ratio_invcov.diagonal()))
    
    
    chi2_realizations = []
    for k in range(100):
        d_realization = np.random.multivariate_normal(theory_ratios, np.linalg.inv(ratio_invcov))
        chi2_data = np.dot(diff,  np.dot(ratio_invcov,  diff))
        diff_realization = d_realization - theory_ratios
        chi2_realization = np.dot(diff_realization, np.dot(ratio_invcov, diff_realization))
        chi2_realizations.append(chi2_realization)
    chi2_realizations = np.array(chi2_realizations)
    print ('boia ',len(chi2_realizations[chi2_realizations>chi2_data]),len(chi2_realizations))
    
        
     
    info_to_save = dict()
    info_to_save['p_v'] = len(chi2_realizations[chi2_realizations>chi2_data])*1./len(chi2_realizations)
    print ('pre sabvin')
    save_obj('./dummy/{0}'.format(config['output_info']),info_to_save)
    print ('after sabvin')
    # un-compressed DVs
    

    return 0