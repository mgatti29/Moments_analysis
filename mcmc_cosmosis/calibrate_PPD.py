
'''
import pickle
import glob
import numpy as np

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



resume = dict()
resume['chi2_realization']  = []
resume['chi2_data']= []
resume['chi2_data0']= []
resume['theory_dprime'] = []
resume['theory_d'] = []
resume['obs_dprime'] = []
resume['dprime_realization'] = []
resume['w'] = []
resume['p_v'] = []
import glob

files = glob.glob('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/dummy_3_2/resume_3_2_*')
for file in files:
    m = load_obj(file.strip('.pkl'))
    for ii in range(len(m['chi2_realization'])):
        resume['chi2_realization'].append(m['chi2_realization'][ii] )
        resume['chi2_data'].append(m['chi2_data'][ii] )
        resume['chi2_data0'].append(m['chi2_data0'][ii] )
        resume['theory_dprime'].append(m['theory_dprime'][ii] )
        resume['obs_dprime'].append(m['obs_dprime'][ii] )
        resume['dprime_realization'].append(m['dprime_realization'][ii] )
        resume['theory_d'].append(m['theory_d'][ii] )
        resume['w'].append(m['w'][ii] )
        p_v = len(m['chi2_realization'][ii][m['chi2_realization'][ii]>m['chi2_data'][ii] ])*1./len(m['chi2_realization'][ii])
        resume['p_v'].append(p_v)
        
resume['chi2_realization'] = np.array(resume['chi2_realization'])
resume['dprime_realization'] = np.array(resume['dprime_realization'])
resume['w'] = np.array(resume['w'])
resume['obs_dprime'] = np.array(resume['obs_dprime'])
resume['theory_dprime'] = np.array(resume['theory_dprime'])
resume['theory_d'] = np.array(resume['theory_d'])
resume['chi2_data'] = np.array(resume['chi2_data'])
resume['chi2_data0'] = np.array(resume['chi2_data0'])

resume['p_v'] = np.array(resume['p_v'])

save_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_3_2',resume)
res = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_3_2')


ndraws = 4000

files_3 = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/test_PPD3')
#files_2 = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/test_PPD2')


nstep_chain = len(resume['theory_dprime'][:,0])
ix = np.argmax(resume['w'])

mu = np.hstack([resume['theory_d'][ix],resume['theory_dprime'][ix]])
uu = np.random.multivariate_normal(mu,files_3['cov12'],ndraws)
inv_cov_d = np.linalg.inv(files_3['cov_1'])
inv_cov_dprime = np.linalg.inv(files_3['cov_2'])
cov_dprime = files_3['cov_2']
cov_d = files_3['cov_1']
theory_dprime_v = resume['theory_dprime']
theory_d_v = resume['theory_d']
c12t = files_3['cov12_cross'].transpose()
c12 = files_3['cov12_cross']

chi_data0 = resume['chi2_data0']
wv = resume['w']


pp =[]

agents = 40 #numero di processi
numero = range(ndraws)
from multiprocessing import Pool,sharedctypes
from functools import partial
from contextlib import closing
        

        
        
def runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0):
    p_ = []
    w_ =[]
    obs_d = uu[k,:15]
    obs_dprime = uu[k,15:]

    for j in range(nstep_chain):
        
        theory_dprime = theory_dprime_v[j,:]
        theory_d = theory_d_v[j,:]

        
        mean_dprime_conditioned = theory_dprime + np.dot(c12, np.dot(inv_cov_d, obs_d-theory_d))
        cov_dprime_conditioned = cov_dprime - np.dot(c12, np.dot(inv_cov_d,c12t ))
    
        chi2_realizations = []
        for k in range(10):
            dprime_realization = np.random.multivariate_normal(mean_dprime_conditioned, cov_dprime_conditioned)
        

        
            diff_data = obs_dprime - theory_dprime
            chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
            diff_realization = dprime_realization - theory_dprime
            chi2_realization = np.dot(diff_realization, np.dot(inv_cov_dprime, diff_realization))
            chi2_realizations.append(chi2_realization)
        chi2_realizations = np.array(chi2_realizations)


        diff_data = obs_dprime - theory_dprime
        chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
        
        diff_data0 = obs_d - theory_d
        chi2_data0 = np.dot(diff_data0, np.dot(inv_cov_d, diff_data0))
    
        w = np.exp(-chi2_data0)/np.exp(-chi_data0[j])*wv[j]
        #exp(-^chi2 (sim-theo))/exp(-^chi2 (obs-theo))
        p_v = len(chi2_realizations[chi2_realizations>chi2_data])*1./len(chi2_realizations)
    
        w_.append(w)
        p_.append(p_v)
    w_ = np.array(w_)
    p_ = np.array(p_)
    mask = (w_==w_)& (p_==p_) & (~np.isinf(w_))& (~np.isinf(p_))

    return np.sum((p_*w_)[mask])/np.sum(w_[mask])
    
#for k in range(5):
#        pp.append(runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0))
with closing(Pool(processes=agents)) as pool:
        pp.append(pool.map(partial(runit,uu=uu,theory_dprime_v=theory_dprime_v,theory_d_v=theory_d_v,c12=c12,c12t=c12t,inv_cov_d=inv_cov_d,  inv_cov_dprime=inv_cov_dprime,wv=wv,chi_data0=chi_data0),numero))

save_obj('calibrated_3_2',pp)







'''





















'''

import pickle
import glob
import numpy as np

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



resume = dict()
resume['chi2_realization']  = []
resume['chi2_data']= []
resume['chi2_data0']= []
resume['theory_dprime'] = []
resume['theory_d'] = []
resume['obs_dprime'] = []
resume['dprime_realization'] = []
resume['w'] = []
resume['p_v'] = []
import glob

files = glob.glob('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/dummy_2_3/resume_2_3_*')
for file in files:
    m = load_obj(file.strip('.pkl'))
    for ii in range(len(m['chi2_realization'])):
        resume['chi2_realization'].append(m['chi2_realization'][ii] )
        resume['chi2_data'].append(m['chi2_data'][ii] )
        resume['chi2_data0'].append(m['chi2_data0'][ii] )
        resume['theory_dprime'].append(m['theory_dprime'][ii] )
        resume['obs_dprime'].append(m['obs_dprime'][ii] )
        resume['dprime_realization'].append(m['dprime_realization'][ii] )
        resume['theory_d'].append(m['theory_d'][ii] )
        resume['w'].append(m['w'][ii] )
        p_v = len(m['chi2_realization'][ii][m['chi2_realization'][ii]>m['chi2_data'][ii] ])*1./len(m['chi2_realization'][ii])
        resume['p_v'].append(p_v)
        
resume['chi2_realization'] = np.array(resume['chi2_realization'])
resume['dprime_realization'] = np.array(resume['dprime_realization'])
resume['w'] = np.array(resume['w'])
resume['obs_dprime'] = np.array(resume['obs_dprime'])
resume['theory_dprime'] = np.array(resume['theory_dprime'])
resume['theory_d'] = np.array(resume['theory_d'])
resume['chi2_data'] = np.array(resume['chi2_data'])
resume['chi2_data0'] = np.array(resume['chi2_data0'])

resume['p_v'] = np.array(resume['p_v'])

save_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_2_3',resume)
res = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_2_3')


ndraws = 4000

files_3 = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/test_PPD2')
#files_2 = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/test_PPD2')


nstep_chain = len(resume['theory_dprime'][:,0])
ix = np.argmax(resume['w'])

mu = np.hstack([resume['theory_d'][ix],resume['theory_dprime'][ix]])
uu = np.random.multivariate_normal(mu,files_3['cov12'],ndraws)
inv_cov_d = np.linalg.inv(files_3['cov_1'])
inv_cov_dprime = np.linalg.inv(files_3['cov_2'])
cov_dprime = files_3['cov_2']
cov_d = files_3['cov_1']
theory_dprime_v = resume['theory_dprime']
theory_d_v = resume['theory_d']
c12t = files_3['cov12_cross'].transpose()
c12 = files_3['cov12_cross']

chi_data0 = resume['chi2_data0']
wv = resume['w']


pp =[]

agents = 40 #numero di processi
numero = range(ndraws)
from multiprocessing import Pool,sharedctypes
from functools import partial
from contextlib import closing
        

        
        
def runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0):
    p_ = []
    w_ =[]
    obs_d = uu[k,:15]
    obs_dprime = uu[k,15:]

    for j in range(nstep_chain):
        
        theory_dprime = theory_dprime_v[j,:]
        theory_d = theory_d_v[j,:]

        
        mean_dprime_conditioned = theory_dprime + np.dot(c12, np.dot(inv_cov_d, obs_d-theory_d))
        cov_dprime_conditioned = cov_dprime - np.dot(c12, np.dot(inv_cov_d,c12t ))
    
        chi2_realizations = []
        for k in range(10):
            dprime_realization = np.random.multivariate_normal(mean_dprime_conditioned, cov_dprime_conditioned)
        

        
            diff_data = obs_dprime - theory_dprime
            chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
            diff_realization = dprime_realization - theory_dprime
            chi2_realization = np.dot(diff_realization, np.dot(inv_cov_dprime, diff_realization))
            chi2_realizations.append(chi2_realization)
        chi2_realizations = np.array(chi2_realizations)


        diff_data = obs_dprime - theory_dprime
        chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
        
        diff_data0 = obs_d - theory_d
        chi2_data0 = np.dot(diff_data0, np.dot(inv_cov_d, diff_data0))
    
        w = np.exp(-chi2_data0)/np.exp(-chi_data0[j])*wv[j]
        #exp(-^chi2 (sim-theo))/exp(-^chi2 (obs-theo))
        p_v = len(chi2_realizations[chi2_realizations>chi2_data])*1./len(chi2_realizations)
    
        w_.append(w)
        p_.append(p_v)
    w_ = np.array(w_)
    p_ = np.array(p_)
    mask = (w_==w_)& (p_==p_) & (~np.isinf(w_))& (~np.isinf(p_))

    return np.sum((p_*w_)[mask])/np.sum(w_[mask])
    
#for k in range(5):
#        pp.append(runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0))
with closing(Pool(processes=agents)) as pool:
        pp.append(pool.map(partial(runit,uu=uu,theory_dprime_v=theory_dprime_v,theory_d_v=theory_d_v,c12=c12,c12t=c12t,inv_cov_d=inv_cov_d,  inv_cov_dprime=inv_cov_dprime,wv=wv,chi_data0=chi_data0),numero))

save_obj('calibrated_2_3',pp)
'''

'''
binx = 3


import pickle
import glob
import numpy as np

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



resume = dict()
resume['chi2_realization']  = []
resume['chi2_data']= []
resume['chi2_data0']= []
resume['theory_dprime'] = []
resume['theory_d'] = []
resume['obs_dprime'] = []
resume['dprime_realization'] = []
resume['w'] = []
resume['p_v'] = []
import glob

files = glob.glob('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/dummy_{0}no/resume_{0}no_*'.format(binx))
for file in files:
    m = load_obj(file.strip('.pkl'))
    for ii in range(len(m['chi2_realization'])):
        resume['chi2_realization'].append(m['chi2_realization'][ii] )
        resume['chi2_data'].append(m['chi2_data'][ii] )
        resume['chi2_data0'].append(m['chi2_data0'][ii] )
        resume['theory_dprime'].append(m['theory_dprime'][ii] )
        resume['obs_dprime'].append(m['obs_dprime'][ii] )
        resume['dprime_realization'].append(m['dprime_realization'][ii] )
        resume['theory_d'].append(m['theory_d'][ii] )
        resume['w'].append(m['w'][ii] )
        p_v = len(m['chi2_realization'][ii][m['chi2_realization'][ii]>m['chi2_data'][ii] ])*1./len(m['chi2_realization'][ii])
        resume['p_v'].append(p_v)
        
resume['chi2_realization'] = np.array(resume['chi2_realization'])
resume['dprime_realization'] = np.array(resume['dprime_realization'])
resume['w'] = np.array(resume['w'])
resume['obs_dprime'] = np.array(resume['obs_dprime'])
resume['theory_dprime'] = np.array(resume['theory_dprime'])
resume['theory_d'] = np.array(resume['theory_d'])
resume['chi2_data'] = np.array(resume['chi2_data'])
resume['chi2_data0'] = np.array(resume['chi2_data0'])

resume['p_v'] = np.array(resume['p_v'])

save_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_{0}no'.format(binx),resume)
res = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_{0}no'.format(binx))


ndraws = 4000

files_3 = load_obj('/project/projectdirs/des/mgatti/Moments_analysis/test_PPD_{0}no'.format(binx))
#files_2 = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/test_PPD2')


nstep_chain = len(resume['theory_dprime'][:,0])
ix = np.argmax(resume['w'])

mu = np.hstack([resume['theory_d'][ix],resume['theory_dprime'][ix]])
uu = np.random.multivariate_normal(mu,files_3['cov12'],ndraws)
inv_cov_d = np.linalg.inv(files_3['cov_1'])
inv_cov_dprime = np.linalg.inv(files_3['cov_2'])
cov_dprime = files_3['cov_2']
cov_d = files_3['cov_1']
theory_dprime_v = resume['theory_dprime']
theory_d_v = resume['theory_d']
c12t = files_3['cov12_cross'].transpose()
c12 = files_3['cov12_cross']

chi_data0 = resume['chi2_data0']
wv = resume['w']


pp =[]

agents = 40 #numero di processi
numero = range(ndraws)
from multiprocessing import Pool,sharedctypes
from functools import partial
from contextlib import closing
        

        
        
def runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0):
    p_ = []
    w_ =[]

    obs_d = uu[k,:theory_d_v.shape[1]]
    obs_dprime = uu[k,theory_d_v.shape[1]:]

    for j in range(nstep_chain):
        
        theory_dprime = theory_dprime_v[j,:]
        theory_d = theory_d_v[j,:]

        
        mean_dprime_conditioned = theory_dprime + np.dot(c12, np.dot(inv_cov_d, obs_d-theory_d))
        cov_dprime_conditioned = cov_dprime - np.dot(c12, np.dot(inv_cov_d,c12t ))
    
        chi2_realizations = []
        for k in range(10):
            dprime_realization = np.random.multivariate_normal(mean_dprime_conditioned, cov_dprime_conditioned)
        

        
            diff_data = obs_dprime - theory_dprime
            chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
            diff_realization = dprime_realization - theory_dprime
            chi2_realization = np.dot(diff_realization, np.dot(inv_cov_dprime, diff_realization))
            chi2_realizations.append(chi2_realization)
        chi2_realizations = np.array(chi2_realizations)


        diff_data = obs_dprime - theory_dprime
        chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
        
        diff_data0 = obs_d - theory_d
        chi2_data0 = np.dot(diff_data0, np.dot(inv_cov_d, diff_data0))
    
        w = np.exp(-chi2_data0)/np.exp(-chi_data0[j])*wv[j]
        #exp(-^chi2 (sim-theo))/exp(-^chi2 (obs-theo))
        p_v = len(chi2_realizations[chi2_realizations>chi2_data])*1./len(chi2_realizations)
    
        w_.append(w)
        p_.append(p_v)
    w_ = np.array(w_)
    p_ = np.array(p_)
    mask = (w_==w_)& (p_==p_) & (~np.isinf(w_))& (~np.isinf(p_))

    return np.sum((p_*w_)[mask])/np.sum(w_[mask])
    
#for k in range(5):
#        pp.append(runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0))
with closing(Pool(processes=agents)) as pool:
        pp.append(pool.map(partial(runit,uu=uu,theory_dprime_v=theory_dprime_v,theory_d_v=theory_d_v,c12=c12,c12t=c12t,inv_cov_d=inv_cov_d,  inv_cov_dprime=inv_cov_dprime,wv=wv,chi_data0=chi_data0),numero))

save_obj('calibrated_{0}no'.format(binx),pp)
'''


'''


binx = 'small'


import pickle
import glob
import numpy as np

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



resume = dict()
resume['chi2_realization']  = []
resume['chi2_data']= []
resume['chi2_data0']= []
resume['theory_dprime'] = []
resume['theory_d'] = []
resume['obs_dprime'] = []
resume['dprime_realization'] = []
resume['w'] = []
resume['p_v'] = []
import glob

files = glob.glob('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/dummy_small/resume_{0}no_*'.format(binx))

for file in files:
    m = load_obj(file.strip('.pkl'))
    for ii in range(len(m['chi2_realization'])):
        resume['chi2_realization'].append(m['chi2_realization'][ii] )
        resume['chi2_data'].append(m['chi2_data'][ii] )
        resume['chi2_data0'].append(m['chi2_data0'][ii] )
        resume['theory_dprime'].append(m['theory_dprime'][ii] )
        resume['obs_dprime'].append(m['obs_dprime'][ii] )
        resume['dprime_realization'].append(m['dprime_realization'][ii] )
        resume['theory_d'].append(m['theory_d'][ii] )
        resume['w'].append(m['w'][ii] )
        p_v = len(m['chi2_realization'][ii][m['chi2_realization'][ii]>m['chi2_data'][ii] ])*1./len(m['chi2_realization'][ii])
        resume['p_v'].append(p_v)
        
resume['chi2_realization'] = np.array(resume['chi2_realization'])
resume['dprime_realization'] = np.array(resume['dprime_realization'])
resume['w'] = np.array(resume['w'])
resume['obs_dprime'] = np.array(resume['obs_dprime'])
resume['theory_dprime'] = np.array(resume['theory_dprime'])
resume['theory_d'] = np.array(resume['theory_d'])
resume['chi2_data'] = np.array(resume['chi2_data'])
resume['chi2_data0'] = np.array(resume['chi2_data0'])

resume['p_v'] = np.array(resume['p_v'])

save_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_{0}no'.format(binx),resume)
res = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_{0}no'.format(binx))


ndraws = 4000

files_3 = load_obj('/project/projectdirs/des/mgatti/Moments_analysis/test_small_large'.format(binx))
#files_2 = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/test_PPD2')


nstep_chain = len(resume['theory_dprime'][:,0])
ix = np.argmax(resume['w'])

mu = np.hstack([resume['theory_d'][ix],resume['theory_dprime'][ix]])
uu = np.random.multivariate_normal(mu,files_3['cov12'],ndraws)
inv_cov_d = np.linalg.inv(files_3['cov_1'])
inv_cov_dprime = np.linalg.inv(files_3['cov_2'])
cov_dprime = files_3['cov_2']
cov_d = files_3['cov_1']
theory_dprime_v = resume['theory_dprime']
theory_d_v = resume['theory_d']
c12t = files_3['cov12_cross'].transpose()
c12 = files_3['cov12_cross']

chi_data0 = resume['chi2_data0']
wv = resume['w']


pp =[]

agents = 40 #numero di processi
numero = range(ndraws)
from multiprocessing import Pool,sharedctypes
from functools import partial
from contextlib import closing
        

        
        
def runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0):
    p_ = []
    w_ =[]

    obs_d = uu[k,:theory_d_v.shape[1]]
    obs_dprime = uu[k,theory_d_v.shape[1]:]

    for j in range(nstep_chain):
        
        theory_dprime = theory_dprime_v[j,:]
        theory_d = theory_d_v[j,:]

        
        mean_dprime_conditioned = theory_dprime + np.dot(c12, np.dot(inv_cov_d, obs_d-theory_d))
        cov_dprime_conditioned = cov_dprime - np.dot(c12, np.dot(inv_cov_d,c12t ))
    
        chi2_realizations = []
        for k in range(10):
            dprime_realization = np.random.multivariate_normal(mean_dprime_conditioned, cov_dprime_conditioned)
        

        
            diff_data = obs_dprime - theory_dprime
            chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
            diff_realization = dprime_realization - theory_dprime
            chi2_realization = np.dot(diff_realization, np.dot(inv_cov_dprime, diff_realization))
            chi2_realizations.append(chi2_realization)
        chi2_realizations = np.array(chi2_realizations)


        diff_data = obs_dprime - theory_dprime
        chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
        
        diff_data0 = obs_d - theory_d
        chi2_data0 = np.dot(diff_data0, np.dot(inv_cov_d, diff_data0))
    
        w = np.exp(-chi2_data0)/np.exp(-chi_data0[j])*wv[j]
        #exp(-^chi2 (sim-theo))/exp(-^chi2 (obs-theo))
        p_v = len(chi2_realizations[chi2_realizations>chi2_data])*1./len(chi2_realizations)
    
        w_.append(w)
        p_.append(p_v)
    w_ = np.array(w_)
    p_ = np.array(p_)
    mask = (w_==w_)& (p_==p_) & (~np.isinf(w_))& (~np.isinf(p_))

    return np.sum((p_*w_)[mask])/np.sum(w_[mask])
    
#for k in range(5):
#        pp.append(runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0))
with closing(Pool(processes=agents)) as pool:
        pp.append(pool.map(partial(runit,uu=uu,theory_dprime_v=theory_dprime_v,theory_d_v=theory_d_v,c12=c12,c12t=c12t,inv_cov_d=inv_cov_d,  inv_cov_dprime=inv_cov_dprime,wv=wv,chi_data0=chi_data0),numero))

save_obj('calibrated_small_large'.format(binx),pp)


'''









binx = 'large'


import pickle
import glob
import numpy as np

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



resume = dict()
resume['chi2_realization']  = []
resume['chi2_data']= []
resume['chi2_data0']= []
resume['theory_dprime'] = []
resume['theory_d'] = []
resume['obs_dprime'] = []
resume['dprime_realization'] = []
resume['w'] = []
resume['p_v'] = []
import glob

files = glob.glob('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/dummy_large/resume_{0}no_*'.format(binx))

for file in files:
    m = load_obj(file.strip('.pkl'))
    for ii in range(len(m['chi2_realization'])):
        resume['chi2_realization'].append(m['chi2_realization'][ii] )
        resume['chi2_data'].append(m['chi2_data'][ii] )
        resume['chi2_data0'].append(m['chi2_data0'][ii] )
        resume['theory_dprime'].append(m['theory_dprime'][ii] )
        resume['obs_dprime'].append(m['obs_dprime'][ii] )
        resume['dprime_realization'].append(m['dprime_realization'][ii] )
        resume['theory_d'].append(m['theory_d'][ii] )
        resume['w'].append(m['w'][ii] )
        p_v = len(m['chi2_realization'][ii][m['chi2_realization'][ii]>m['chi2_data'][ii] ])*1./len(m['chi2_realization'][ii])
        resume['p_v'].append(p_v)
        
resume['chi2_realization'] = np.array(resume['chi2_realization'])
resume['dprime_realization'] = np.array(resume['dprime_realization'])
resume['w'] = np.array(resume['w'])
resume['obs_dprime'] = np.array(resume['obs_dprime'])
resume['theory_dprime'] = np.array(resume['theory_dprime'])
resume['theory_d'] = np.array(resume['theory_d'])
resume['chi2_data'] = np.array(resume['chi2_data'])
resume['chi2_data0'] = np.array(resume['chi2_data0'])

resume['p_v'] = np.array(resume['p_v'])

save_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_{0}no'.format(binx),resume)
res = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/dummy/resume_{0}no'.format(binx))


ndraws = 4000

files_3 = load_obj('/project/projectdirs/des/mgatti/Moments_analysis/test_large_small'.format(binx))
#files_2 = load_obj('/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/test_PPD2')


nstep_chain = len(resume['theory_dprime'][:,0])
ix = np.argmax(resume['w'])

mu = np.hstack([resume['theory_d'][ix],resume['theory_dprime'][ix]])
uu = np.random.multivariate_normal(mu,files_3['cov12'],ndraws)
inv_cov_d = np.linalg.inv(files_3['cov_1'])
inv_cov_dprime = np.linalg.inv(files_3['cov_2'])
cov_dprime = files_3['cov_2']
cov_d = files_3['cov_1']
theory_dprime_v = resume['theory_dprime']
theory_d_v = resume['theory_d']
c12t = files_3['cov12_cross'].transpose()
c12 = files_3['cov12_cross']

chi_data0 = resume['chi2_data0']
wv = resume['w']


pp =[]

agents = 40 #numero di processi
numero = range(ndraws)
from multiprocessing import Pool,sharedctypes
from functools import partial
from contextlib import closing
        

        
        
def runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0):
    p_ = []
    w_ =[]

    obs_d = uu[k,:theory_d_v.shape[1]]
    obs_dprime = uu[k,theory_d_v.shape[1]:]

    for j in range(nstep_chain):
        
        theory_dprime = theory_dprime_v[j,:]
        theory_d = theory_d_v[j,:]

        
        mean_dprime_conditioned = theory_dprime + np.dot(c12, np.dot(inv_cov_d, obs_d-theory_d))
        cov_dprime_conditioned = cov_dprime - np.dot(c12, np.dot(inv_cov_d,c12t ))
    
        chi2_realizations = []
        for k in range(10):
            dprime_realization = np.random.multivariate_normal(mean_dprime_conditioned, cov_dprime_conditioned)
        

        
            diff_data = obs_dprime - theory_dprime
            chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
            diff_realization = dprime_realization - theory_dprime
            chi2_realization = np.dot(diff_realization, np.dot(inv_cov_dprime, diff_realization))
            chi2_realizations.append(chi2_realization)
        chi2_realizations = np.array(chi2_realizations)


        diff_data = obs_dprime - theory_dprime
        chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime, diff_data))
        
        diff_data0 = obs_d - theory_d
        chi2_data0 = np.dot(diff_data0, np.dot(inv_cov_d, diff_data0))
    
        w = np.exp(-chi2_data0)/np.exp(-chi_data0[j])*wv[j]
        #exp(-^chi2 (sim-theo))/exp(-^chi2 (obs-theo))
        p_v = len(chi2_realizations[chi2_realizations>chi2_data])*1./len(chi2_realizations)
    
        w_.append(w)
        p_.append(p_v)
    w_ = np.array(w_)
    p_ = np.array(p_)
    mask = (w_==w_)& (p_==p_) & (~np.isinf(w_))& (~np.isinf(p_))

    return np.sum((p_*w_)[mask])/np.sum(w_[mask])
    
#for k in range(5):
#        pp.append(runit(k,uu,theory_dprime_v,theory_d_v,c12,c12t,inv_cov_d,inv_cov_dprime,wv,chi_data0))
with closing(Pool(processes=agents)) as pool:
        pp.append(pool.map(partial(runit,uu=uu,theory_dprime_v=theory_dprime_v,theory_d_v=theory_d_v,c12=c12,c12t=c12t,inv_cov_d=inv_cov_d,  inv_cov_dprime=inv_cov_dprime,wv=wv,chi_data0=chi_data0),numero))

save_obj('calibrated_large_small'.format(binx),pp)