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



def data_compression(redshift_config,bins,scales,dc_dict,emu_config,inv_cov,physical_scale_cut,Nz_mean):
    Nz0 = load_Nz(["0.2_0.43","0.43_0.63","0.63_0.9","0.9_1.3","0.2_1.3"],redshift_config["path"])

    yy = make_theory(Nz0,bins,scales,dc_dict["p0"],emu_config,physical_scale_cut,Nz_mean)
    #print len(yy)
    transf_matrix = np.zeros((len(dc_dict["p0"].keys()),len(yy)))

    for jj,key in enumerate(dc_dict["p0"].keys()):
        
        p1_dict = copy.copy(dc_dict["p0"])
    
        ddh = np.zeros((len(yy),4))
        der = np.zeros(len(yy))

        
        p1_dict[key] = dc_dict["p0"][key]- 2*dc_dict["d0"][key]
        ddh[:,0] = make_theory(Nz0,bins,scales,p1_dict,emu_config,physical_scale_cut,Nz_mean)
        p1_dict[key] = dc_dict["p0"][key]- dc_dict["d0"][key]
        ddh[:,1] = make_theory(Nz0,bins,scales,p1_dict,emu_config,physical_scale_cut,Nz_mean)
        p1_dict[key] =dc_dict["p0"][key] +dc_dict["d0"][key]
        ddh[:,2] = make_theory(Nz0,bins,scales,p1_dict,emu_config,physical_scale_cut,Nz_mean)
        p1_dict[key] = dc_dict["p0"][key]+ 2*dc_dict["d0"][key]
        ddh[:,3] = make_theory(Nz0,bins,scales,p1_dict,emu_config,physical_scale_cut,Nz_mean)

        der = -(- ddh[:,0] +8* ddh[:,1]-8* ddh[:,2]+ ddh[:,3])/(12*dc_dict["d0"][key])
    
        #print der.shape,inv_cov.shape
        transf_matrix[jj,:]= np.matmul(der,inv_cov).T
  
    return transf_matrix 

def setup_runs(moments_conf,mcmc_config,params,priors,scales,bins_dictionary,config,cov_config,emu_config,redshift_config,dc_dict = {"flag":False},ph={"ph":None,"Nn":None},hartlap={"Hartlap":True,"DS":True,"Sellentin":False}):
        """
        dc_dict contains the info for the compression
        ph the scales for the cut
        hartlap specifies which corrections to the covariance one wants to include
        """
    
        for cov_c in config['covariances']:
            for ii,tomo_c in enumerate(config['bins']):
                tomo_combo_l = tomo_c.replace("+",'_').replace(" ",'')
                mute_dict = dict()
                mute_dict.update({"bins":bins_dictionary[tomo_c]})
                mute_dict.update({"config":config})
                mute_dict.update({"mcmc_config":mcmc_config})
                mute_dict.update({"emu_config":emu_config})
                mute_dict.update({"scales":scales})
                mute_dict.update({"ph":ph["ph"]})
                mute_dict.update({"Nz_mean":ph["Nn"]})
                mute_dict.update({'Nz' : redshift_config})
                #mute_dict.update({'Nz' :load_Nz(redshift_config["bins_to_load"],redshift_config["path"])})
                physical_scale_cut=ph["ph"]
                Nz_mean=ph["Nn"]
                
                if config["data_vector"] == "theory":
                    mute_dict.update({'y_obs' :make_theory(mute_dict["Nz"],bins_dictionary[tomo_c],scales,params,emu_config,physical_scale_cut)})
                else:
                    mute = load_obj(config["data_vector"] )
                    mute_dict.update({'y_obs' :make_vector(bins_dictionary[tomo_c],redshift_config["bins_to_load"],mute,scales,physical_scale_cut,Nz_mean)})
        
        
                cov,_,_ = make_covariance(bins_dictionary[tomo_c],cov_c,cov_config["flask"],cov_config["numb_of_real"],scales,physical_scale_cut,Nz_mean)

                inv_cov = linalg.inv(cov["cov"]*config['boost']*config['boost'])
        
        
                try:
                    N_p = cov_config["numb_of_real_subst"]["numb_of_real"]
                except:
                    N_p=cov_config["numb_of_real"]-1
                p_p=len(mute_dict['y_obs'])
                n_pars = len(np.array(priors.keys())[~np.array(["multi"in x for x in priors.keys()])])
                
                f_hartlap=1.
                if hartlap["Hartlap"]:
                        
                    f_hartlap=f_hartlap*(N_p-1.)/(N_p-p_p-2.)
                        
                if hartlap["DS"]:
                    fds = 1 + (p_p-n_pars)*(N_p-p_p-2.)/((N_p-p_p-1.)*(N_p-p_p-4.))
                    f_hartlap=f_hartlap*fds
                
                mute_dict.update({"Sellentin":{"S":hartlap["Sellentin"],"Ns":cov_config["numb_of_real"]}})
                
                mute_dict.update({'cov' :cov["cov"]*config['boost']*config['boost']})
                mute_dict.update({'inv_cov' :inv_cov/f_hartlap})
                # setup priors **********************************
                mute_dict.update({"prior_functions":setup_priors(priors,params)})
    
                if dc_dict["flag"]:
                    t_matrix = data_compression(redshift_config,bins_dictionary[tomo_c],scales,dc_dict,emu_config,inv_cov/f_hartlap,physical_scale_cut,Nz_mean)
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
                    
                    try:

                        if dc_dict["red_taka"]:
                            _,vv3,_ = make_covariance(bins_dictionary[tomo_c],dc_dict["covv_taka_label"],dc_dict["taka_cov"]["flask"],dc_dict["taka_cov"]["numb_of_real"],scales,physical_scale_cut,Nz_mean)
                            vv4 = np.zeros((dc_dict["taka_cov"]["numb_of_real"],t_matrix.shape[0]))
                            for df in range(dc_dict["taka_cov"]["numb_of_real"]):
                                vv4[df,:] = np.matmul(t_matrix,vv3[df,:])
                            
                            cov_dict = covariance_jck (vv4.T,dc_dict["taka_cov"]["numb_of_real"],"bootstrap")
                        
                            inv_cov = linalg.inv(cov_dict["cov"]*config['boost']*config['boost'])
                            
                            try:
                                N_p = dc_dict["taka_cov"]["numb_of_real_subst"]["numb_of_real"]
                            except:
                                N_p=dc_dict["taka_cov"]["numb_of_real"]-1
                    
                    
                            p_p=new_c.shape[0]
                            
                            f_hartlap=1.
                            if hartlap["Hartlap"]:
                        
                                f_hartlap=f_hartlap*(N_p-1.)/(N_p-p_p-2.)
                        
                            if hartlap["DS"]:
                                fds = 1 + (p_p-n_pars)*(N_p-p_p-2.)/((N_p-p_p-1.)*(N_p-p_p-4.))
                                f_hartlap=f_hartlap*fds

                            mute_dict.update({'cov' :cov_dict ["cov"]*config['boost']*config['boost']})
                            mute_dict.update({'inv_cov' :inv_cov/f_hartlap})

                        
                        
                    except:
                        pass
                    mute_dict["y_obs"] = np.matmul(t_matrix,mute_dict["y_obs"])
                    mute_dict["transf_matrix"] = t_matrix
                    mute_dict["dc"] =True
                else:
                    mute_dict["dc"] = False
                    
                moments_conf.update({'{0}_{1}_{2}'.format(tomo_combo_l,cov_c,config["label"]): mute_dict})

        return moments_conf
    
    
    

def make_vector(bins_dict,bins,vectorss,scales,physical_scale_cut=None,Nz_mean=None):
    
        cosmo = {'omega_M_0':0.3,
             'omega_lambda_0':1-0.3,
              'omega_k_0': 0.0,
              'omega_b_0' : 0.047,
               'h':0.7,
               'sigma_8' : 0.82}
        
    
        theta = vectorss['theta']
        
        count = 0
        for bi in bins_dict['bins2']:
        
            zzm=0.
            min_theta_rp=0.
            if physical_scale_cut:
                for bx in bi:
                    zzm+=Nz_mean[bx-1]
                min_theta_rp=(physical_scale_cut[0]/((1.+zzm/2.)*cd.comoving_distance(zzm/2.,**cosmo)*(2*math.pi)/360)*60)
            mask = np.array(scales['scales_2'])>min_theta_rp
            lent = len(np.array(scales['scales_2'])[mask])
            count+=lent
        for bi in bins_dict['bins3']:
            zzm=0.
            min_theta_rp=0.
            if physical_scale_cut:
                for bx in bi:
                    zzm+=Nz_mean[bx-1]
                min_theta_rp=(physical_scale_cut[1]/((1.+zzm/3.)*cd.comoving_distance(zzm/3.,**cosmo)*(2*math.pi)/360)*60)
            mask = np.array(scales['scales_3'])>min_theta_rp
            lent = len(np.array(scales['scales_3'])[mask])
            count+=lent
        vector = np.zeros(count)



        count = 0
        for bi in bins_dict['bins2']:
            zzm=0.
            min_theta_rp=0.
            if physical_scale_cut:
                for bx in bi:
                    zzm+=Nz_mean[bx-1]
                min_theta_rp=(physical_scale_cut[0]/((1.+zzm/2.)*cd.comoving_distance(zzm/2.,**cosmo)*(2*math.pi)/360)*60)
            masku = np.array(scales['scales_2'])>min_theta_rp
            
            lent = len(np.array(scales['scales_2'])[masku])
            binx = str(bins[bi[0]-1])+"_"+str(bins[bi[1]-1])
            mask = np.in1d(theta,np.array(scales['scales_2'])[masku])
            vector[count:(count+lent)]=vectorss[binx][mask]
            count+=lent
        for bi in bins_dict['bins3']:
            zzm=0.
            min_theta_rp=0.
            if physical_scale_cut:
                for bx in bi:
                    zzm+=Nz_mean[bx-1]
                min_theta_rp=(physical_scale_cut[1]/((1.+zzm/3.)*cd.comoving_distance(zzm/3.,**cosmo)*(2*math.pi)/360)*60)
            masku = np.array(scales['scales_3'])>min_theta_rp
            lent = len(np.array(scales['scales_3'])[masku])
            mask = np.in1d(theta,np.array(scales['scales_3'])[masku])
            binx = str(bins[bi[0]-1])+"_"+str(bins[bi[1]-1])+"_"+str(bins[bi[2]-1])
            vector[count:(count+lent)]=vectorss[binx][mask]
            count+=lent
        return vector
    
#setup priors

def lnprob_trunc_norm(x, mean, bounds, C):
    if np.any(x < bounds[:,0]) or np.any(x > bounds[:,1]):
        return -np.inf
    else:
        return -0.5*(x-mean).dot(inv(C)).dot(x-mean)
    
    
    
    
def lnprob_trunc_norm(x, mean,  C):
    import scipy
    bounds=np.array([[mean[i]-8*xc,mean[i]+8*xc] for i,xc in enumerate(np.sqrt(C.diagonal()))])
    #print (bounds)
    #print (x)
   # print (mean)
    if np.any(x < bounds[:,0]) or np.any(x > bounds[:,1]):
        return -np.inf
    else:
        #return -(x-mean).dot(scipy.linalg.inv(C)).dot(x-mean)
        return scipy.stats.multivariate_normal.logpdf(x,mean,(C))
        #return scipy.stats.multivariate_normal(x,mean,(scipy.linalg.inv(C))).logpdf(x)
    
    
    
    
def setup_priors(priors,params):
    initial_values = []
    list_var = []
    prior_function =dict()
    
    for key in priors.keys():
        if priors[key][0]=="uniform":
            prior_function.update({key: {'kind': 'uniform', 'weight': 1,
                                   'kwargs': {'loc': priors[key][1][0], 'scale': priors[key][1][1]-priors[key][1][0]}}}) 
        elif priors[key][0]=="gaussian":
            prior_function.update({key: {'kind': 'truncnorm', 'weight': 1,
                                    'kwargs': {'a': -8.0, 'b': 8.0,
                                               'loc': priors[key][1][0], 'scale': priors[key][1][1]}}})
            
        elif "multi" in key:
                
                prior_function.update({key:{'params':priors[key][0],'mean':priors[key][1],"cov":priors[key][2]}})
    
        
    
    from scipy import stats
    for key in prior_function.keys():
        
        if "multi" in key:
            prior_function[key] = prior_function[key]
        else:
            prior_function[key]= getattr(stats, prior_function[key]['kind'])(**prior_function[key]['kwargs'])
  


    return prior_function

    

def make_covariance(mapp, num_real, scales, bins_dictionary):
    '''
    This routine create a covariance matrix from a list of FLASK moments.
    mapp: list of flask moments.
    num_real: number of realisations
    scales: smoothing scales
    bins_dictionary: it informs the code about the correlation and scales that needs to be included.
    
    
    '''
    
    # fiducial cosmology for the scale cut conversion.
    cosmo = {'omega_M_0':0.3,
             'omega_lambda_0':1-0.3,
              'omega_k_0': 0.0,
              'omega_b_0' : 0.047,
               'h':0.7,
               'sigma_8' : 0.82}
    
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
                if bins_dictionary[key]['scale_cut'] == 'physical':
                    # we take the mean of the <z> of the tomographic bins involved. We use this redshift
                    # to convert a cut from physical scales to angular scales.
                    for bx in bi:
                        zzm += Nz_mean[bx-1]
                    min_theta_rp=(physical_scale_cut[0]/((1.+zzm/len(bi))*cd.comoving_distance(zzm/len(bi),**cosmo)*(2*math.pi)/360)*60)
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
                    if bins_dictionary[key]['scale_cut'] == 'physical':
                        # we take the mean of the <z> of the tomographic bins involved. We use this redshift
                        # to convert a cut from physical scales to angular scales.
                        for bx in bi:
                            zzm += Nz_mean[bx-1]
                        min_theta_rp=(physical_scale_cut[0]/((1.+zzm/len(bi))*cd.comoving_distance(zzm/len(bi),**cosmo)*(2*math.pi)/360)*60)
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
                if bins_dictionary[key]['scale_cut'] == 'physical':
                    # we take the mean of the <z> of the tomographic bins involved. We use this redshift
                    # to convert a cut from physical scales to angular scales.
                    for bx in bi:
                        zzm += Nz_mean[bx-1]
                    min_theta_rp=(physical_scale_cut[0]/((1.+zzm/len(bi))*cd.comoving_distance(zzm/len(bi),**cosmo)*(2*math.pi)/360)*60)
                    mask_scales =np.array(scales)>min_theta_rp
            except:
                pass
            mask = np.in1d(mapp[0].conf['smoothing_scales'],np.array(scales)[mask_scales])
            lent = len(mapp[0].conf['smoothing_scales'][mask])
            err_dict_plot[key+'_'+binx] = cov_dict['err'][count:(count+lent)]
            count +=lent
    return cov_dict,vector,err_dict_plot



def make_theo_plot(theo,bins_dict,scales,bins,physical_scale_cut=None,Nz_mean=None):
    cosmo = {'omega_M_0':0.3,
             'omega_lambda_0':1-0.3,
              'omega_k_0': 0.0,
              'omega_b_0' : 0.047,
               'h':0.7,
               'sigma_8' : 0.82}
                     

    theo = np.array(theo)
    theo2_plot = dict()
    count = 0
    for bi in bins_dict['bins2']:
        zzm=0.
        min_theta_rp=-5.
        if physical_scale_cut:
            for bx in bi:
                zzm+=Nz_mean[bx-1]
            min_theta_rp=(physical_scale_cut[0]/((1.+zzm/2.)*cd.comoving_distance(zzm/2.,**cosmo)*(2*math.pi)/360)*60)

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
            min_theta_rp=(physical_scale_cut[1]/((1.+zzm/3.)*cd.comoving_distance(zzm/3.,**cosmo)*(2*math.pi)/360)*60)

        binx = str(bins[bi[0]-1])+"_"+str(bins[bi[1]-1])+"_"+str(bins [bi[2]-1])
        mask = np.array(scales['scales_3'])>min_theta_rp
        lent = len(np.array(scales['scales_3'])[mask])
        theo2_plot[binx] = theo[count:(count+lent)]
        count+=lent
    return theo2_plot



def setup_initial_values(info_p):
    priors = info_p
    initial_values = []
    list_var = []
    if 'Omega_m' in info_p:
        initial_values.append(priors["Omega_m"].rvs())
        list_var.append("Omega_m")
    if  'Sigma_8' in info_p:
        initial_values.append(priors["Sigma_8"].rvs())
        list_var.append("Sigma_8") 
    if 'Omega_b' in info_p:
        initial_values.append(priors["Omega_b"].rvs())
        list_var.append("Omega_b") 
    if 'n_s' in info_p:
        initial_values.append(priors["n_s"].rvs())
        list_var.append("n_s") 
    if 'h100' in info_p:
        initial_values.append(priors["h100"].rvs())
        list_var.append("h100") 
        



    if 'm1' in info_p:
        initial_values.append(priors["m1"].rvs())
        list_var.append("m1")
    if 'm2' in info_p:
        initial_values.append(priors["m2"].rvs())
        list_var.append("m2")
    if 'm3' in info_p:
        initial_values.append(priors["m3"].rvs())
        list_var.append("m3")
    if 'm4' in info_p:
        initial_values.append(priors["m4"].rvs())
        list_var.append("m4")
    if 'm5' in info_p:
        initial_values.append(priors["m5"].rvs())
        list_var.append("m5")
        
        


             
    if 'A_IA' in info_p:
        initial_values.append(priors["A_IA"].rvs())
        list_var.append("A_IA") 
    if 'alpha' in info_p:
        initial_values.append(priors["alpha"].rvs())
        list_var.append("alpha") 
                

    if 'dz1' in info_p:
        initial_values.append(priors["dz1"].rvs())
        list_var.append("dz1")   
    if 'dz2' in info_p:
        initial_values.append(priors["dz2"].rvs())
        list_var.append("dz2")   
    if 'dz3' in info_p:
        initial_values.append(priors["dz3"].rvs())
        list_var.append("dz3")   
    if 'dz4' in info_p:
        initial_values.append(priors["dz4"].rvs())
        list_var.append("dz4")   
    if 'dz5' in info_p:
        initial_values.append(priors["dz5"].rvs())
        list_var.append("dz5")   
        
    return initial_values,list_var



def update_progress(progress,elapsed_time=0,starting_time=0):
    import time
    import timeit
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))



    if progress*100>1. and elapsed_time>0 :
        remaining=((elapsed_time-starting_time)/progress)*(1.-progress)
        text = "\rPercent: [{0}] {1:.2f}% {2}  - elapsed time: {3} - estimated remaining time: {4}".format( "#"*block + "-"*(barLength-block), progress*100, status,time.strftime('%H:%M:%S',time.gmtime(elapsed_time-starting_time)),time.strftime('%H:%M:%S',time.gmtime(remaining)))
    else:
        text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()