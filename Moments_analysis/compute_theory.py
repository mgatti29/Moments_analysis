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

import pickle
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f)
        f.close()
    return mute


def compute_theory(Nz0,scales2,scales3,bins_array_2,bins_array_3,om,s8,m1=0.,m2=0.,m3=0.,m4=0.,m5=0.,dz1=0.,dz2=0.,dz3=0.,dz4=0.,dz5=0.,A0=0.,alpha0=0.,z0=0.67,ns=0.97,h100=0.7,ob=0.046,emu_dict=None,physical_scale_cut=None,Nz_mean = None):
                import cosmolopy.distance as cd
                import cosmolopy
                import scipy
                from scipy.interpolate import interp1d
                import timeit
                lib2=ctypes.cdll.LoadLibrary("../mcmc/mcmc_code_repo/double_cycle.so")
                start = timeit.default_timer()
                m = [m1,m2,m3,m4,m5]
                cosmo = {'omega_M_0':om, 
                     'omega_lambda_0':1-om,
                     'omega_k_0': 0.0, 
                     'omega_b_0' : ob,
                     'h':h100,
                     'sigma_8' : s8,
                     'n': ns}
                z = emu_dict['z_EMU']
                svd_train =  emu_dict['svd_train'] 
                def lensing_kernel23_fast(dave_lr,dchislr,dave_hr,dchishr,zhr,Nz):
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
                    return weigh




                growth = (cosmolopy.perturbation.fgrowth(z,cosmo['omega_M_0']))
                growth = (cosmolopy.perturbation.fgrowth(z,cosmo['omega_M_0']))
                dchis = 1./cd.e_z(z,**cosmo)#/(100*cosmo['h'])
                dchism = 1./cd.e_z(z+0.5*(z[1]-z[0]),**cosmo)#/(100*cosmo['h'])
    
                # interpolation of the distance
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



                end= timeit.default_timer()
                #print "cosmology: ",end-start
    
                start = timeit.default_timer()
                dz = [dz1,dz2,dz3,dz4,dz5]
    
                # load nz ***************************
                Nz = []
                Qz = []
                # Nz_mean = []


    
                for i,dz0 in enumerate(dz):
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
                    
                    

                    #qmm1a = om*lensing_kernel23(dave_lr,dchislr,dave_hr,dchishr,zhr,nzm_m)
                    qmm1a = om*np.array(list(lensing_kernel23_fast(dave_lr,dchislr,dave_hr,dchishr,zhr,nzm_m)))
        
                    #print qmm1a,qmm1ab
                    fq = interp1d(zhr, qmm1a*1.5)    
                    qmm1=  fq(z[1:-1])
                    qmm = np.zeros(len(qmm1)+2)
                    qmm[0],qmm[1:-1]=qmm1a[0]*1.5,qmm1[:]
        
        
                    Qz.append(np.array(qmm))


                end = timeit.default_timer()
                #print "redshift: ",end-start

                #print (Nz_mean)
                # make vector:

                x00 = []
                for sm in scales2:
                    x00.append([sm,om,s8])

                # SVD ***************
                if 1==1:

                    input_cosmology =[np.array([s8,om,ob,ns,h100])]
                    smoothing_scales_emu = np.array([0.0,2.0,3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.])


                    d2_array = []
                    d3_array = []
                    d3_array_c = []

                    for sm in np.arange(len(smoothing_scales_emu))[np.in1d(smoothing_scales_emu,scales2)]:
                        dim = len(svd_train[sm][0][-1])
                        v0 = []
                        for i in range(dim):
                            YY_std = svd_train[sm][0][4][i]
                            YY_mean = svd_train[sm][0][3][i]
                            YY_in = (svd_train[sm][0][2][i]-YY_mean)/YY_std
                            v0.append((svd_train[sm][0][-1][i].predict(YY_in, input_cosmology)[0])*YY_std+YY_mean)
                        v0 = np.array(v0).reshape(dim)
                        d2_array.append(np.exp(np.array(np.dot(svd_train[sm][0][0][:,:dim] * svd_train[sm][0][1][:dim], v0))))

                    for sm in np.arange(len(smoothing_scales_emu))[np.in1d(smoothing_scales_emu,scales3)]:
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
                        for sm in np.arange(len(smoothing_scales_emu))[np.in1d(smoothing_scales_emu,scales3)]:
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

                        
                        
                    d2 = np.array(d2_array).T
                    

                end1 = timeit.default_timer()

                #print "svd: ",end1-end

                m2 = dict()
                m3 = dict()
                len_models=0
                for i,binx2 in enumerate(bins_array_2):
                    len_models+= len(scales2)
                    IA = A0*(((1+z[1:])/(1+z0))**alpha0)*0.0134*om/growth[1:]
                    IA1 = A0*(((1+z[1:])/(1+z0))**alpha0)*0.0134*om/growth[1:]

                    weight1 = Qz[binx2[0]-1][1:]*dist[1:]- Nz[binx2[0]-1][0][1:]/(dchis[1:])*IA 
                    weight2 = Qz[binx2[1]-1][1:]*dist[1:]- Nz[binx2[1]-1][0][1:]/(dchis[1:])*IA1
                    mute = weight1*weight2
                    m2.update({str(binx2):2*np.dot((d2[1:,:]).T,mute*dchis[1:]*(z[1]-z[0]))})
                for i,binx3 in enumerate(bins_array_3):
                    len_models+= len(scales3)
                    IA = A0*(((1+z[1:])/(1+z0))**alpha0)*0.0134*om/growth[1:]
                    IA1 = A0*(((1+z[1:])/(1+z0))**alpha0)*0.0134*om/growth[1:]
                    IA2 = A0*(((1+z[1:])/(1+z0))**alpha0)*0.0134*om/growth[1:]

                    #weight = this->lens_kernels[bins3[3*jj]-1].weight_at_comoving_distance(w)*w -this->lens_kernels[bins3[3*jj]-1].nz_at_comoving_distance(w)*IA*w ;
                    #weight1 = this->lens_kernels[bins3[3*jj+1]-1].weight_at_comoving_distance(w)*w -this->lens_kernels[bins3[3*jj+1]-1].nz_at_comoving_distance(w)*IA1*w;
                    #weight2 = this->lens_kernels[bins3[3*jj+2]-1].weight_at_comoving_distance(w)*w -this->lens_kernels[bins3[3*jj+2]-1].nz_at_comoving_distance(w)*IA2*w;


                    weight1 = Qz[binx3[0]-1][1:]*dist[1:] - Nz[binx3[0]-1][0][1:]/(dchis[1:])*IA 
                    weight2 = Qz[binx3[1]-1][1:]*dist[1:] - Nz[binx3[1]-1][0][1:]/(dchis[1:])*IA1 
                    weight3 = Qz[binx3[2]-1][1:]*dist[1:] - Nz[binx3[2]-1][0][1:]/(dchis[1:])*IA2 
                    mute = weight1*weight2*weight3
                    m3.update({str(binx3):6*np.dot((d3[1:,:]).T,mute*dchis[1:]*(z[1]-z[0]))})
                end2 = timeit.default_timer()
                
                
        
        
                vector = np.empty(len_models)
                vectorm = np.empty(len_models)
                count=0
                
                #make bins:
                bins =[]
                for bi in bins_array_2:
                    bins.append(bi)
                for bi in bins_array_3:
                    bins.append(bi)
                
                
                for binx in bins_array_2:
                    m_factor = 1.
                    cxx = 0
                    for u in (((str(binx).split('['))[1]).split(']')[0]).split(','):
                        m_factor = m_factor*(1+m[np.int(u)-1])
                        cxx+=1
                    for ss,scale in enumerate(scales2):
                        vectorm[count] = m_factor
                        if cxx==3:
                            vector[count] = m_factor*m3[str(binx)][ss]
                        if cxx==2:
                            vector[count] = m_factor*m2[str(binx)][ss]
                        count+=1
                        
                for binx in bins_array_3:
                    m_factor = 1.
                    cxx = 0
                    for u in (((str(binx).split('['))[1]).split(']')[0]).split(','):
                        m_factor = m_factor*(1+m[np.int(u)-1])
                        cxx+=1
                    for ss,scale in enumerate(scales3):
                        vectorm[count] = m_factor
                        if cxx==3:
                            vector[count] = m_factor*m3[str(binx)][ss]
                        if cxx==2:
                            vector[count] = m_factor*m2[str(binx)][ss]
                        count+=1
                        
                cosmo = {'omega_M_0':0.3,
                     'omega_lambda_0':1-0.3,
                     'omega_k_0': 0.0, 
                     'omega_b_0' : 0.047,
                     'h':0.7,
                     'sigma_8' : 0.82}
                     
                count = 0
                count1 = 0
                if physical_scale_cut:
                    vector = []
                   
                    for binx in bins_array_2:
                        zzm=0.
                        
                        m_factor = 1.
                        cxx = 0
                        for u in (((str(binx).split('['))[1]).split(']')[0]).split(','):
                            m_factor = m_factor*(1+m[np.int(u)-1])
                            zzm+=Nz_mean[np.int(u)-1]
                            cxx+=1
                        min_theta_rp=(physical_scale_cut[0]/((1.+zzm/2.)*cd.comoving_distance(zzm/2.,**cosmo)*(2*math.pi)/360)*60)
                        #print (min_theta_rp)
                        for ss,scale in enumerate(scales2):
                            count+=1
                            if scale>min_theta_rp:
                                count1+=1
                                #vectorm[count] = m_factor
                                if cxx==3:
                                    vector.append(m_factor*m3[str(binx)][ss])
                                if cxx==2:
                                    vector.append(m_factor*m2[str(binx)][ss])

                    for binx in bins_array_3:
                        m_factor = 1.
                        zzm=0.
                        cxx = 0
                        for u in (((str(binx).split('['))[1]).split(']')[0]).split(','):
                                m_factor = m_factor*(1+m[np.int(u)-1])
                                zzm+=Nz_mean[np.int(u)-1]
                                cxx+=1
                        min_theta_rp=(physical_scale_cut[1]/((1.+zzm/3.)*cd.comoving_distance(zzm/3.,**cosmo)*(2*math.pi)/360)*60)
                        #print (min_theta_rp)
                        for ss,scale in enumerate(scales3):
                            count+=1
                            if scale>min_theta_rp:
                                count1+=1
                                #vectorm[count] = m_factor
                                if cxx==3:
                                    vector.append(m_factor*m3[str(binx)][ss])
                                if cxx==2:
                                    vector.append(m_factor*m2[str(binx)][ss])
                                

                    vector=np.array(vector)
 
                return vector
            
            
def make_theory(Nz0,bins_dict,scales,params,emu_dict,ph=None,Nz_mean=None):
    if 'Omega_m' in params.keys():
        om = params["Omega_m"]
    else:
        om = 0.3
    if 'Sigma_8' in params.keys():
        s8 = params["Sigma_8"]
    else:
        s8 = 0.8
    if 'n_s' in params.keys():
        ns = params["n_s"]
    else:
        ns = 0.97
    if 'Omega_b' in params.keys():
        ob = params["Omega_b"]
    else:
        ob = 0.046
    if 'h100' in params.keys():
        h100 = params["h100"]
    else:
        h100 = 0.7
    if 'dz1' in params.keys():
        dz1 = params["dz1"]
    else:
        dz1 = 0.       
    if 'dz2' in params.keys():
        dz2 = params["dz2"]
    else:
        dz2 = 0.
    if 'dz3' in params.keys():
        dz3 = params["dz3"]
    else:
        dz3 = 0.       
    if 'dz4' in params.keys():
        dz4 = params["dz4"]
    else:
        dz4 = 0.
    if 'dz5' in params.keys():
        dz5 = params["dz5"]
    else:
        dz5 = 0.

    if 'm1' in params.keys():
        m1 = params["m1"]
    else:
        m1 = 0.       
    if 'm2' in params.keys():
        m2 = params["m2"]
    else:
        m2 = 0.
    if 'm3' in params.keys():
        m3 = params["m3"]
    else:
        m3 = 0.       
    if 'm4' in params.keys():
        m4 = params["m4"]
    else:
        m4 = 0.
    if 'm5' in params.keys():
        m5 = params["m5"]
    else:
        m5 = 0.

    if 'A_IA' in params.keys():
        A_IA = params["A_IA"]
    else:
        A_IA = 0.
    if 'alpha0' in params.keys():
        alpha0 = params["alpha0"]
    else:
        alpha0 = 0.

    vector = compute_theory(Nz0,scales['scales_2'],scales['scales_3'],bins_dict['bins2'],bins_dict['bins3'],om,s8,m1,m2,m3,m4,m5,dz1,dz2,dz3,dz4,dz5,A0=A_IA,alpha0=alpha0,z0=0.67,ns=ns,h100=h100,ob=ob,emu_dict=emu_dict,physical_scale_cut=ph,Nz_mean=Nz_mean)
    return vector 
        
def lnlike(p):
    p=np.array(p)

    om = p[list_var=="Omega_m"]
    s8 = p[list_var=="Sigma_8"]
    ob = p[list_var=="Omega_b"]
    ns = p[list_var=="n_s"]
    h100 = p[list_var=="h100"]

    if "dz1" in list_var:
        dz1= p[list_var=="dz1"][0]
    else:
        dz1=0.
    if "dz2" in list_var:
        dz2= p[list_var=="dz2"][0]
    else:
        dz2=0.
    if "dz3" in list_var:
        dz3= p[list_var=="dz3"][0]
    else:
        dz3=0.
    if "dz4" in list_var:
        dz4= p[list_var=="dz4"][0]
    else:
        dz4=0.
    if "dz5" in list_var:
        dz5= p[list_var=="dz5"][0]
    else:
        dz5=0.
        
        
    if "b1" in list_var:
        b1= p[list_var=="b1"][0]
    else:
        b1=0.
    if "b2" in list_var:
        b2= p[list_var=="b2"][0]
    else:
        b2=0.
    if "b3" in list_var:
        b3= p[list_var=="b3"][0]
    else:
        b3=0.
    if "b4" in list_var:
        b4= p[list_var=="b4"][0]
    else:
        b4=0.
    if "b5" in list_var:
        b5= p[list_var=="b5"][0]
    else:
        b5=0.
        
    if "I_IA" in list_var:
        A0= p[list_var=="I_IA"][0]
    else:
        A0=0.
    if "alpha" in list_var:
        alpha0= p[list_var=="alpha"][0]
    else:
        alpha0=0.
    z0=0.65
    #print(len_models,theta,om,s8,b1,b2,b3,b4,b5,dz1,dz2,dz3,dz4,dz5,A0,alpha0,z0)
    y = np.zeros(len(y_obs)) #give_back_theory(len_models,theta,om,s8,b1,b2,b3,b4,b5,dz1,dz2,dz3,dz4,dz5,A0,alpha0,z0,ns,h100,ob,GP=True)

    
    
    w = y-y_obs
    #make vector out of it
    
    chi2 = np.matmul(w,np.matmul(np.linalg.inv(cov_obs),w))

    return -0.5 * chi2

def lnprior(p):
    #logprior = [prior.logpdf(pi) * wi for pi, prior, wi in zip(p, self.prior_function, self.prior_weight)]
    logprior =1 
    return logprior

def log_prob(p):
    #self.nsteps += 1
    #update_progress(float(self.nsteps / self.nwalkers / (self.nburnin + self.nrun)),timeit.default_timer(),self.start)
    #prior
    lp = np.sum(lnprior(p))
    if not np.isfinite(lp):
        return -np.inf
    #likelihood
    ll = np.sum(lnlike(p))
    return lp + ll

    
def load_Nz(bins_theory,path_redshift):

    
    # Load Nz ********
    Nz0 = []
    for i,binx in enumerate(bins_theory):
        #path = '../../moments/moment_computation_fast_2/multi_z_Buzzard_g1g2_{0}.txt'.format(binx)
        path = '{1}_{0}.txt'.format(binx,path_redshift)
        mute = np.loadtxt(path)
        ddz = (mute[1,0]+mute[0,0])*0.5
        z_m,f_m = np.zeros(len(mute[:,0])+1),np.zeros(len(mute[:,0])+1)
        z_m[1:]=mute[:,0]+ddz
        f_m[1:]=mute [:,1]
        #print z_m
        fz = interp1d(z_m,f_m)
        Nz0.append(fz)
    return Nz0
    

    
