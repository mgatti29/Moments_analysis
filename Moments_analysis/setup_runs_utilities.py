# prepare covariance
import numpy as np
import sys
import math
import scipy
from scipy import linalg
import cosmolopy.distance as cd
import cosmolopy
from .compute_theory import *
import copy

def covariance_jck(vec, jk_r, type_cov):
    """
    Estimate covariance matrix using jackknife or bootstrap method.

    Args:
        vec (array): Input data matrix.
        jk_r (int): Number of jackknife regions or bootstrap samples.
        type_cov (str): Type of covariance estimation. Can be 'jackknife' or 'bootstrap'.

    Returns:
        dict: Dictionary containing covariance matrix ('cov'), error vector ('err'), correlation matrix ('corr'), and mean vector ('mean').
    """
    len_w = len(vec[:, 0])
    jck = len(vec[0, :])

    err = np.zeros(len_w)
    mean = np.zeros(len_w)
    cov = np.zeros((len_w, len_w))
    corr = np.zeros((len_w, len_w))

    mean = np.zeros(len_w)
    for i in range(jk_r):
        mean += vec[:, i]
    mean = mean / jk_r

    if type_cov == 'jackknife':
        fact = (jk_r - 1.) / (jk_r)
    elif type_cov == 'bootstrap':
        fact = 1. / (jk_r - 1)

    for i in range(len_w):
        for j in range(len_w):
            for k in range(jck):
                cov[i, j] += (vec[i, k] - mean[i]) * (vec[j, k] - mean[j]) * fact

    for ii in range(len_w):
        err[ii] = np.sqrt(cov[ii, ii])

    for i in range(len_w):
        for j in range(len_w):
            corr[i, j] = cov[i, j] / (np.sqrt(cov[i, i] * cov[j, j]))

    return {'cov': cov,
            'err': err,
            'corr': corr,
            'mean': mean}


def data_compression(redshift_config,bins,scales,dc_dict,emu_config,inv_cov):
    '''
    In order to compress a given data vector, you need to be able to compute the derivative
    of your data vector wrt all the parameters in your analysis (cosmological + nuisance).
    The output will be a matrix of length (N_params,len_uncompressed_DV).
    Then you just need to do:
    t_matrix # this the compression matrix
    
    compressed_cov = np.matmul(t_matrix,np.matmul(uncompressed_cov,t_matrix.T))
    compressed_DV = np.matmul(t_matrix,uncompressed_DV)
                    
    '''
    
    # I need to load a few things to be able to compute the moments now.
    
    # Load Nz
    Nz0 = load_Nz(redshift_config["bins_to_load"],redshift_config["path"])

    # compute theory at some fiducial cosmology
    yy,_ = compute_theory(Nz0, bins, scales,dc_dict["p0"],emu_config)#

    # Define the compression matrix . Note tat dc_dict["p0"] is a dictionary of all the parameters
    # at their fiducial values (such that you can compute the derivatives)
    
    transf_matrix = np.zeros((len(dc_dict["p0"].keys()),len(yy)))

    u0 = []
    [[ u0.append(j) for j in u] for u in (bins['bins'])]
    unique_bins =  np.unique(np.array(u0))-1
    # number of parameters:
    numb = 0
    for key in dc_dict["d0"].keys():
        try:
            len(dc_dict["d0"][key])
            numb += len(unique_bins)
        except:
            numb+=1
    
    transf_matrix = np.zeros((numb,len(yy)))

    count = 0
    for key in dc_dict["p0"].keys():
        
        p1_dict = copy.copy(dc_dict["p0"])
    
        ddh = np.zeros((len(yy),4))
        der = np.zeros(len(yy))

        try:
            if len(dc_dict["p0"][key]) > 1:
                
                for jj in range(len(dc_dict["p0"][key])):
                    if jj in unique_bins:
                        p1_dict[key][jj] = dc_dict["p0"][key][jj]- 2.*dc_dict["d0"][key][jj]
                        u,_ = compute_theory(Nz0, bins, scales,p1_dict,emu_config)
                        ddh[:,0] = u
                        p1_dict[key][jj] = dc_dict["p0"][key][jj]- dc_dict["d0"][key][jj]
                        u,_ = compute_theory(Nz0, bins, scales,p1_dict,emu_config)
                        ddh[:,1] = u
                        p1_dict[key][jj] =dc_dict["p0"][key][jj] +dc_dict["d0"][key][jj]
                        u,_ = compute_theory(Nz0, bins, scales,p1_dict,emu_config)
                        ddh[:,2] = u
                        p1_dict[key][jj] = dc_dict["p0"][key][jj]+ 2.*dc_dict["d0"][key][jj]
                        u,_ = compute_theory(Nz0, bins, scales,p1_dict,emu_config)
                        ddh[:,3] = u
                        der = -(- ddh[:,0] +8* ddh[:,1]-8* ddh[:,2]+ ddh[:,3])/(12.*dc_dict["d0"][key][jj])
                    
                        #print der.shape,inv_cov.shape
                        transf_matrix[count,:]= np.matmul(der,inv_cov).T
                        count+=1
        except:
                p1_dict[key] = dc_dict["p0"][key]- 2*dc_dict["d0"][key]
                u,_ = compute_theory(Nz0, bins, scales,p1_dict,emu_config)

                ddh[:,0] = u
                p1_dict[key] = dc_dict["p0"][key]- dc_dict["d0"][key]
                u,_ = compute_theory(Nz0, bins, scales,p1_dict,emu_config)
                ddh[:,1] = u
                p1_dict[key] =dc_dict["p0"][key] +dc_dict["d0"][key]
                u,_ = compute_theory(Nz0, bins, scales,p1_dict,emu_config)
                ddh[:,2] = u
                p1_dict[key] = dc_dict["p0"][key]+ 2*dc_dict["d0"][key]
                u,_ = compute_theory(Nz0, bins, scales,p1_dict,emu_config)
                ddh[:,3] = u
                der = -(- ddh[:,0] +8* ddh[:,1]-8* ddh[:,2]+ ddh[:,3])/(12*dc_dict["d0"][key])
            
                #print der.shape,inv_cov.shape
                transf_matrix[count,:]= np.matmul(der,inv_cov).T
                count+=1
    return transf_matrix 

def setup_runs(moments_conf,mcmc_config,priors,scales,bins_dictionary,config,cov_config,emu_config,redshift_config,dc_dict = {"flag":False},hartlap={"Hartlap":True,"DS":True,"Sellentin":False}, alternative_cov = {'doit':False,'cov':None}):
        """
        Set up  for mcmc runs.

        Args:
            moments_conf (dict): Dictionary to store the configuration for moments analysis.
            mcmc_config (dict): Configuration for MCMC.
            priors (dict): Prior information.
            scales (dict): Scales information.
            bins_dictionary (dict): Dictionary containing bin information.
            config (dict): General configuration information.
            cov_config (dict): Covariance configuration information.
            emu_config (dict): Emulator configuration information.
            redshift_config (dict): Redshift configuration information.
            dc_dict (dict, optional): Dictionary containing compression information. Defaults to {"flag": False}.
            hartlap (dict, optional): Hartlap correction information. Defaults to {"Hartlap": True, "DS": True, "Sellentin": False}.
            alternative_cov (dict, optional): Alternative covariance information. Defaults to {'doit': False, 'cov': None}.

        Returns:
            dict: Updated moments_conf dictionary.
        """
        for cov_c in config['covariances']:
            for ii,tomo_c in enumerate(config['bins']):
                tomo_combo_l = tomo_c.replace("+",'_').replace(" ",'')
                mute_dict = dict()
                mute_dict.update({"bins":bins_dictionary[tomo_c][cov_c]})
                mute_dict.update({"config":config})
                mute_dict.update({"mcmc_config":mcmc_config})
                mute_dict.update({"emu_config":emu_config})
                mute_dict.update({"scales":scales})
                mute_dict.update({'Nz' : redshift_config})
                #pritn (bins_dictionary[tomo_c][cov_c].keys())
                mute_dict.update({'y_obs' :make_vector(bins_dictionary[tomo_c][cov_c],redshift_config["bins_to_load"],load_obj(config["data_vector"] ),scales,bins_dictionary[tomo_c][cov_c]['scale_cut'],bins_dictionary[tomo_c][cov_c]['Nz_mean'])})
                u0 = []
                [[ u0.append(j) for j in u] for u in (bins_dictionary[tomo_c][cov_c]['bins'])]
                unique_bins =  np.unique(np.array(u0))-1

            
                
                cov,dv_j_cov,_ = make_covariance(cov_config["flask"],cov_config["numb_of_real"],scales,bins_dictionary[tomo_c])

                mute_dict.update({'cov_uncompr':cov}) 
                if alternative_cov['doit']:
                    Acov,vv3,_ = make_covariance(alternative_cov['cov']["flask"],alternative_cov['cov']["numb_of_real"],scales,bins_dictionary[tomo_c])
                    mute_dict.update({'Acov_uncompr':Acov}) 

                    
                inv_cov = linalg.inv(cov["cov"]*config['boost']*config['boost'])
        
        
                try:
                    N_p = cov_config["numb_of_real_subst"]["numb_of_real"]
                except:
                    N_p=cov_config["numb_of_real"]-1
                p_p=len(mute_dict['y_obs'])
                
                n_pars = 0
                for key in dc_dict["d0"].keys():
                    try:
                        len(dc_dict["d0"][key])
                        n_pars += len(unique_bins)
                    except:
                        n_pars+=1
    
  
                f_hartlap=1.
                if hartlap["Hartlap"]:
                        
                    f_hartlap=f_hartlap*(N_p-1.)/(N_p-p_p-2.)
                        
                if hartlap["DS"]:
                    fds = 1 + (p_p-n_pars)*(N_p-p_p-2.)/((N_p-p_p-1.)*(N_p-p_p-4.))
                    f_hartlap=f_hartlap*fds
                
                mute_dict.update({"Sellentin":{"S":hartlap["Sellentin"],"Ns":cov_config["numb_of_real"]}})
                
                mute_dict.update({'cov' :cov["cov"]*config['boost']*config['boost']})
                mute_dict.update({'dv_j_cov':dv_j_cov})
                mute_dict.update({'inv_cov' :inv_cov/f_hartlap})
                # setup priors **********************************
                #mute_dict.update({"prior_functions":setup_priors(priors)})
    
                if dc_dict["flag"]:
                    t_matrix = data_compression(redshift_config,bins_dictionary[tomo_c][cov_c],scales,dc_dict,emu_config,inv_cov/f_hartlap)
                
                    #if alternative_cov['doit']:
                    #    new_c= np.matmul(t_matrix,np.matmul((Acov["cov"]*config['boost']*config['boost']),t_matrix.T))
                    #else:
                    new_c = np.matmul(t_matrix,np.matmul((cov["cov"]*config['boost']*config['boost']),t_matrix.T))
                    
                    inv_cov = linalg.inv(new_c)
                    try:
                        N_p = cov_config["numb_of_real_subst"]["numb_of_real"]
                    except:
                        N_p=cov_config["numb_of_real"]-1
                    
                    p_p=new_c.shape[0]
                    
                    f_hartlap=1.
                    if hartlap["Hartlap"]:
                        
                        f_hartlap=f_hartlap*(N_p-1.)/(N_p-p_p-2.)
                        
                    if hartlap["DS"]:
                        fds = 1 + (p_p-n_pars)*(N_p-p_p-2.)/((N_p-p_p-1.)*(N_p-p_p-4.))
                        f_hartlap=f_hartlap*fds
                
                    
                    mute_dict.update({'cov' :new_c})
                    mute_dict.update({'inv_cov' :inv_cov/f_hartlap})
                    
                    if alternative_cov['doit']:
                            vv4 = np.zeros((alternative_cov['cov']["numb_of_real"],t_matrix.shape[0]))
                            for df in range(alternative_cov['cov']["numb_of_real"]):
                                vv4[df,:] = np.matmul(t_matrix,vv3[df,:])
                            
                            cov_dict = covariance_jck (vv4.T,alternative_cov['cov']["numb_of_real"],"bootstrap")
                        
                            inv_cov = linalg.inv(cov_dict["cov"]*config['boost']*config['boost'])
                            
                            N_p=alternative_cov['cov']["numb_of_real"]-1
                    
                    
                            p_p=new_c.shape[0]
                            
                            f_hartlap=1.
                            if hartlap["Hartlap"]:
                        
                                f_hartlap=f_hartlap*(N_p-1.)/(N_p-p_p-2.)
                        
                            if hartlap["DS"]:
                                fds = 1 + (p_p-n_pars)*(N_p-p_p-2.)/((N_p-p_p-1.)*(N_p-p_p-4.))
                                f_hartlap=f_hartlap*fds

                            mute_dict.update({'cov' :cov_dict ["cov"]*config['boost']*config['boost']})
                            print ('update')
                            mute_dict.update({'inv_cov' :inv_cov/f_hartlap})

                    mute_dict["y_obs"] = np.matmul(t_matrix,mute_dict["y_obs"])
                    mute_dict["transf_matrix"] = t_matrix
                    mute_dict["dc"] =True
                else:
                    mute_dict["dc"] = False
                    
                moments_conf.update({'{0}_{1}_{2}'.format(tomo_combo_l,cov_c,config["label"]): mute_dict})

        return moments_conf
 

def make_vector(bins_dict,bins,vectorss,scales,physical_scale_cut=None,Nz_mean=None):
        """
        Create a vector from the provided data.

        Args:
            bins_dict (dict): Dictionary containing bin information.
            bins (list): List of bins.
            vectorss (dict): Data vector.
            scales (array): Array of scales.
            physical_scale_cut (float, optional): Physical scale cut. Defaults to None.
            Nz_mean (array, optional): Mean redshift values. Defaults to None.

        Returns:
            array: Vector created from the data.
        """
    
        # cosmology
        cosmo_scale = {'omega_M_0':0.28 , 
                     'omega_lambda_0':1 - 0.28 ,
                     'omega_k_0': 0.0, 
                     'omega_b_0' : 0.047 ,
                     'h':0.7 ,
                     'sigma_8' :  0.82,
                     'n': 0.97}
        
    
        theta = vectorss['theta']
        count = 0
        for bi in bins_dict['bins']:
            # binx: label of the bins in the format from FLASK
            binx = '_'.join([str(l-1) for l in bi])
            mask_scales = scales == scales
            
            # scale cut ***
            zzm=0.
            min_theta_rp=0.
            if bins_dict['scale_cut'] != None :
                # we take the mean of the <z> of the tomographic bins involved. We use this redshift
                # to convert a cut from physical scales to angular scales.
                for bx in bi:
                    zzm += bins_dict['Nz_mean'][bx-1]
                min_theta_rp=(bins_dict['scale_cut']/((1.+zzm/len(bi))*cd.comoving_distance(zzm/len(bi),**cosmo_scale)*(2*math.pi)/360)*60)
                mask_scales =np.array(scales)>min_theta_rp   
                count += len(np.array(scales[mask_scales]))
            else:
                count += len(np.array(scales[mask_scales]))

        vector = np.zeros(count)



        count = 0
        for bi in bins_dict['bins']:
            # binx: label of the bins in the format from FLASK
            binx = '_'.join([str(l-1) for l in bi])
            mask_scales = scales == scales
            
            # scale cut ***
            zzm=0.
            min_theta_rp=0.
            if bins_dict['scale_cut'] != None :
                # we take the mean of the <z> of the tomographic bins involved. We use this redshift
                # to convert a cut from physical scales to angular scales.
                for bx in bi:
                    zzm += bins_dict['Nz_mean'][bx-1]
                min_theta_rp=(bins_dict['scale_cut']/((1.+zzm/len(bi))*cd.comoving_distance(zzm/len(bi),**cosmo_scale)*(2*math.pi)/360)*60)
                masku =np.array(scales)>min_theta_rp   
                lent = len(np.array(scales)[masku])
                mask = np.in1d(theta,np.array(scales)[masku])
                vector[count:(count+lent)]=vectorss[binx][mask]
                count+=lent
                
        return vector

    

    

def make_covariance(mapp, num_real, scales, bins_dictionary):
    '''
    This routine create a covariance matrix from a list of FLASK moments.
    mapp: list of flask moments.
    num_real: number of realisations
    scales: smoothing scales
    bins_dictionary: it informs the code about the correlation and scales that needs to be included.
    
    
    '''
    
    # fiducial cosmology for the scale cut conversion.
    cosmo_scale = {'omega_M_0':0.28 , 
                     'omega_lambda_0':1 - 0.28 ,
                     'omega_k_0': 0.0, 
                     'omega_b_0' : 0.047 ,
                     'h':0.7 ,
                     'sigma_8' :  0.82,
                     'n': 0.97}
    
    import numpy as np
    
    # setup bins ************************************
    bins = []
    len_v = 0
    
    # this bit appends the bins and apply scale cut if needed
    for key in bins_dictionary.keys():
        for bi in bins_dictionary[key]['bins']:
            bins.append(bi)

            try:
                # scale cut ***
                zzm=0.
                min_theta_rp=0.
                if bins_dictionary[key]['scale_cut'] != None:
                    # we take the mean of the <z> of the tomographic bins involved. We use this redshift
                    # to convert a cut from physical scales to angular scales.
                    for bx in bi:
                        zzm += bins_dictionary[key]['Nz_mean'][bx-1]
                    min_theta_rp=(bins_dictionary[key]['scale_cut']/((1.+zzm/len(bi))*cd.comoving_distance(zzm/len(bi),**cosmo_scale)*(2*math.pi)/360)*60)
                    mask_scales = (np.array(scales)>min_theta_rp) 
                    len_v+= len(np.array(scales[mask_scales]))
                else:
                    len_v += len(np.array(scales))
                    
            except:
                len_v += len(np.array(scales))
    
    # now we can initialise the vector holding the moments from different FLASK simulations.
    vector = np.zeros((num_real,len_v))
    for jk in range(num_real):
        count=0
        for key in bins_dictionary.keys():
            for bi in bins_dictionary[key]['bins']:
                # binx: label of the bins in the format from FLASK
                binx = '_'.join([str(l-1) for l in bi])
                mask_scales = scales == scales
                try:
                    # scale cut ***
                    zzm=0.
                    min_theta_rp=0.
                    if bins_dictionary[key]['scale_cut'] != None:
                        # we take the mean of the <z> of the tomographic bins involved. We use this redshift
                        # to convert a cut from physical scales to angular scales.
                        for bx in bi:
                            zzm += bins_dictionary[key]['Nz_mean'][bx-1]
                        min_theta_rp=(bins_dictionary[key]['scale_cut']/((1.+zzm/len(bi))*cd.comoving_distance(zzm/len(bi),**cosmo_scale)*(2*math.pi)/360)*60)
                        mask_scales =np.array(scales)>min_theta_rp   
                        
                except:
                    pass
                mask = np.in1d(mapp[0].conf['smoothing_scales'],np.array(scales)[mask_scales])
                lent = len(mapp[0].conf['smoothing_scales'][mask])
                vector[jk,count:(count+lent)] = mapp[jk].moments[key][binx][mask]
                count+=lent
                
    cov_dict = covariance_jck(vector.T,num_real,"bootstrap")
    
    # the routine also gives back a dictionary with the diagonal elements of the covariance divided per bin combinations.
    err_dict_plot = dict()
    count = 0
    for key in bins_dictionary.keys():
        for bi in bins_dictionary[key]['bins']:
            # binx: label of the bins in the format from FLASK
            binx = '_'.join([str(l-1) for l in bi])
            mask_scales = scales == scales
            try:
                # scale cut ***
                zzm=0.
                min_theta_rp=0.
                if bins_dictionary[key]['scale_cut'] != None:
                    # we take the mean of the <z> of the tomographic bins involved. We use this redshift
                    # to convert a cut from physical scales to angular scales.
                    for bx in bi:
                        zzm += bins_dictionary[key]['Nz_mean'][bx-1]
                    min_theta_rp=(bins_dictionary[key]['scale_cut']/((1.+zzm/len(bi))*cd.comoving_distance(zzm/len(bi),**cosmo_scale)*(2*math.pi)/360)*60)
                    mask_scales =np.array(scales)>min_theta_rp
            except:
                pass
            mask = np.in1d(mapp[0].conf['smoothing_scales'],np.array(scales)[mask_scales])
            lent = len(mapp[0].conf['smoothing_scales'][mask])
            err_dict_plot[key+'_'+binx] = cov_dict['err'][count:(count+lent)]
            count +=lent
    return cov_dict,vector,err_dict_plot



def make_theo_plot(theo,bins_dict,scales,bins,physical_scale_cut=None,Nz_mean=None):
    """
    Create a  plot based on the provided data.

    Args:
        theo (array): Theoretical data.
        bins_dict (dict): Dictionary containing bin information.
        scales (dict): Dictionary of scales.
        bins (list): List of bins.
        physical_scale_cut (array, optional): Physical scale cut. Defaults to None.
        Nz_mean (array, optional): Mean redshift values. Defaults to None.

    Returns:
        dict: Dictionary containing the theoretical plot.
    """
    
    #setup cosmology
    cosmo_scale = {'omega_M_0':0.28 , 
                     'omega_lambda_0':1 - 0.28 ,
                     'omega_k_0': 0.0, 
                     'omega_b_0' : 0.047 ,
                     'h':0.7 ,
                     'sigma_8' :  0.82,
                     'n': 0.97}
                     

    theo = np.array(theo)
    theo2_plot = dict()
    count = 0
    for bi in bins_dict['bins2']:
        zzm=0.
        min_theta_rp=-5.
        if physical_scale_cut:
            for bx in bi:
                zzm+=Nz_mean[bx-1]
            min_theta_rp=(physical_scale_cut[0]/((1.+zzm/2.)*cd.comoving_distance(zzm/2.,**cosmo_scale)*(2*math.pi)/360)*60)

        binx = str(bins[bi[0]-1])+"_"+str(bins[bi[1]-1])#+"_"+str(bins [bi[2]-1])
        mask = np.array(scales['scales_2'])>min_theta_rp
        lent = len(np.array(scales['scales_2'])[mask])
        theo2_plot[binx] = theo[count:(count+lent)]
        count+=lent
    for bi in bins_dict['bins3']:
        zzm=0.
        min_theta_rp=-5.
        if physical_scale_cut:
            for bx in bi:
                zzm+=Nz_mean[bx-1]
            min_theta_rp=(physical_scale_cut[1]/((1.+zzm/3.)*cd.comoving_distance(zzm/3.,**cosmo_scale)*(2*math.pi)/360)*60)

        binx = str(bins[bi[0]-1])+"_"+str(bins[bi[1]-1])+"_"+str(bins [bi[2]-1])
        mask = np.array(scales['scales_3'])>min_theta_rp
        lent = len(np.array(scales['scales_3'])[mask])
        theo2_plot[binx] = theo[count:(count+lent)]
        count+=lent
    return theo2_plot





