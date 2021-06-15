import numpy as np
import pickle
import xarray as xr
import os
from mpi4py import MPI
import math
import glob
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
def make_values(i,uu,path): 
    
    a = open(path,'w')
    
    a.write('[cosmological_parameters]\n')
    a.write('omega_m      =  {0}\n'.format(uu['cosmological_parametersomega_m'][i]))
    a.write('h0           =  {0}\n'.format(uu['cosmological_parametersh0'][i]))
    a.write('omega_b      =  {0}\n'.format(uu['cosmological_parametersomega_b'][i]))
    a.write('n_s          =  {0}\n'.format(uu['cosmological_parametersn_s'][i]))
    a.write('sigma8_input =  {0}\n'.format(uu['cosmological_parameterssigma8_input'][i]))
    a.write('omnuh2       =  0.00083 \n')
    a.write('massive_nu   =  3\n')
    a.write('massless_nu  =  0.046\n')
    a.write('A_s = 2.215e-9\n')
    a.write('omega_k      =  0.0\n')
    a.write('tau          =  0.0697186\n')
    a.write(';Helium mass fraction.  Needed for Planck \n')
    a.write('yhe          = 0.245341\n')
    a.write('[shear_calibration_parameters]\n')
    a.write('m1 = {0}\n'.format(uu['shear_calibration_parametersm1'][i]))
    a.write('m2 = {0}\n'.format(uu['shear_calibration_parametersm2'][i]))
    a.write('m3 = {0}\n'.format(uu['shear_calibration_parametersm3'][i]))
    a.write('m4 = {0}\n'.format(uu['shear_calibration_parametersm4'][i]))

    a.write('[lens_photoz_errors]\n')
    a.write('bias_1 = 0.\n')
    a.write('bias_2 = 0.\n')
    a.write('bias_3 = 0.\n')
    a.write('bias_4 =  0.0\n')
    a.write('bias_5 =  0.0\n')
    a.write('width_1 = 1.0\n')
    a.write('width_2 = 1.0\n')
    a.write('width_3 = 1.0\n')
    a.write('width_4 = 1.0\n')
    a.write('width_5 = 1.0\n')
    a.write('[bias_lens]\n')
    a.write('b1 = 1.7\n')
    a.write('b2 = 1.7\n')
    a.write('b3 = 1.7\n')
    a.write('b4 = 2.0\n')
    a.write('b5 = 2.0\n')
    a.write('[mag_alpha_lens]\n')
    a.write('alpha_1 =  1.31\n')
    a.write('alpha_2 = -0.52\n')
    a.write('alpha_3 =  0.34\n')
    a.write('alpha_4 =  2.25\n')
    a.write('alpha_5 =  1.97\n')
    a.write('[intrinsic_alignment_parameters]\n')
    a.write('z_piv   =  0.62\n')
    a.write('A1      = {0}\n'.format(uu['intrinsic_alignment_parametersa1'][i]))
    a.write('A2      = 0.0  \n')
    a.write('alpha1  = {0}\n'.format(uu['intrinsic_alignment_parametersalpha1'][i]))
    a.write('alpha2  = 0.0 \n')
    a.write('bias_ta =  0.0 \n')
    a.write('[wl_photoz_errors]\n')
    a.write('bias_1 = {0}\n'.format(uu['wl_photoz_errorsbias_1'][i]))
    a.write('bias_2 = {0}\n'.format(uu['wl_photoz_errorsbias_2'][i]))
    a.write('bias_3 = {0}\n'.format(uu['wl_photoz_errorsbias_3'][i]))
    a.write('bias_4 = {0}\n'.format(uu['wl_photoz_errorsbias_4'][i]))

    a.close()
    '''
    a = open(path,'w')
    a.write('[cosmological_parameters]\n')
    a.write('omega_m      =  0.2856330329\n'.format(uu['cosmological_parametersomega_m'][i]))
    a.write('h0           =  0.5556 \n'.format(uu['cosmological_parametersh0'][i]))
    a.write('omega_b      =  0.03712\n'.format(uu['cosmological_parametersomega_b'][i]))
    a.write('n_s          =  1.057379\n'.format(uu['cosmological_parametersn_s'][i]))
    a.write('sigma8_input =  0.8104452\n'.format(uu['cosmological_parameterssigma8_input'][i]))
    a.write('omnuh2       =  0.00083 \n')
    a.write('massive_nu   =  3\n')
    a.write('massless_nu  =  0.046\n')
    a.write('A_s = 2.215e-9\n')
    a.write('omega_k      =  0.0\n')
    a.write('tau          =  0.0697186\n')
    a.write(';Helium mass fraction.  Needed for Planck \n')
    a.write('yhe          = 0.245341\n')
    a.write('[shear_calibration_parameters]\n')
    a.write('m1 = -0.009802668 \n'.format(uu['shear_calibration_parametersm1'][i]))
    a.write('m2 = -0.015201963 \n'.format(uu['shear_calibration_parametersm2'][i]))
    a.write('m3 = -0.021540926 \n'.format(uu['shear_calibration_parametersm3'][i]))
    a.write('m4 = -0.039804863 \n'.format(uu['shear_calibration_parametersm4'][i]))

    a.write('[lens_photoz_errors]\n')
    a.write('bias_1 = 0.\n')
    a.write('bias_2 = 0.\n')
    a.write('bias_3 = 0.\n')
    a.write('bias_4 =  0.0\n')
    a.write('bias_5 =  0.0\n')
    a.write('width_1 = 1.0\n')
    a.write('width_2 = 1.0\n')
    a.write('width_3 = 1.0\n')
    a.write('width_4 = 1.0\n')
    a.write('width_5 = 1.0\n')
    a.write('[bias_lens]\n')
    a.write('b1 = 1.7\n')
    a.write('b2 = 1.7\n')
    a.write('b3 = 1.7\n')
    a.write('b4 = 2.0\n')
    a.write('b5 = 2.0\n')
    a.write('[mag_alpha_lens]\n')
    a.write('alpha_1 =  1.31\n')
    a.write('alpha_2 = -0.52\n')
    a.write('alpha_3 =  0.34\n')
    a.write('alpha_4 =  2.25\n')
    a.write('alpha_5 =  1.97\n')
    a.write('[intrinsic_alignment_parameters]\n')
    a.write('z_piv   =  0.62\n')
    a.write('A1      = 0. \n'.format(uu['intrinsic_alignment_parametersa1'][i]))
    a.write('A2      = 0.0  \n')
    a.write('alpha1  = 0. \n'.format(uu['intrinsic_alignment_parametersalpha1'][i]))
    a.write('alpha2  = 0.0 \n')
    a.write('bias_ta =  0.0 \n')
    a.write('[wl_photoz_errors]\n')
    a.write('bias_1 = -0.00609 \n'.format(uu['wl_photoz_errorsbias_1'][i]))
    a.write('bias_2 = -0.00437 \n'.format(uu['wl_photoz_errorsbias_2'][i]))
    a.write('bias_3 = -0.00425 \n'.format(uu['wl_photoz_errorsbias_3'][i]))
    a.write('bias_4 = 0.010158 \n'.format(uu['wl_photoz_errorsbias_4'][i]))

    a.close()


    '''
    
    
    
    
    





if __name__ == '__main__':
    '''
    import os
    
    # Read Unblinded 2nd chain ********* ********* ********* ********* ********* ********* ********* ********* ********* *********
    ou = '/global/cscratch1/sd/mgatti/Mass_Mapping/chains_cosmosis/chain_2tomocross_kEkE_UNBlinded.txt'
    #ou = '/project/projectdirs/des/www/y3_chains/3x2pt/final_paper_chains/chain_1x2_0321.txt'



    #if I understood calibration, I should load the 2_3 chain above, and the multiply the weights with some sort of importance sampling
    #to get back to chain_2.
    #weight = np.exp(-chi2_realization)/np.exp(-2_3_chain[chi^2])
    
    
    uu = np.genfromtxt(ou, names = True)
    len_chain  = len(uu['cosmological_parametersomega_m'])
    chunk_size = 30
    number_of_runs = math.ceil(len_chain/chunk_size)
    if not os.path.exists('./dummy/dummy_2_3/'):
        os.mkdir('./dummy/dummy_2_3/')


    def doit(index,chunk_size,len_chain,uu): 
        ranges = [index*chunk_size,min([(index+1)*chunk_size,len_chain])]
        
        chi2_realization = []
        chi2_data = []
        chi2_data0= []
        dprime_realization = []
        theory_dprime = []
        theory_d = []
        obs_dprime = []
        w =[]
        p_v = []
        
        resume = dict()
        if not os.path.exists('./dummy/dummy_2_3/Aresume_2_3_{0}.pkl'.format(index)):
            for i in range(ranges[0],ranges[1]):
                make_values(i,uu,'./dummy/values_dummi_2_3_{0}.ini'.format(i))
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_2_3.ini'
                os.environ["DATAFILE"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits'
                os.environ["DATAFILE_SR"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy'
                os.environ["SCALEFILE"] = 'Y3_params_priors/scales.ini'
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_2_3.ini'
                os.environ["VALUESINCLUDE"] = 'dummy/values_dummi_2_3_{0}.ini'.format(i)
                os.environ["PRIORSINCLUDE"] = 'Y3_params_priors/priors.ini'
                os.environ["output_info"] = 'info_dummy_2_3_{0}'.format(i)
                
         
                os.system('cosmosis Y3_params_priors/params.ini ')

                info_to_save =load_obj('./dummy/info_dummy_2_3_{0}'.format(i))
                chi2_realization.append(info_to_save['chi2_realization'])
                chi2_data.append(info_to_save['chi2_data'])
                dprime_realization.append(info_to_save['dprime_realization'])
                theory_dprime.append(info_to_save['theory_dprime'])
                theory_d.append(info_to_save['theory_d'])
                
                obs_dprime.append(info_to_save['obs_dprime'])
                p_v.append(info_to_save['p_v'])
                chi2_data0.append(info_to_save['chi2_data0'])
                w.append(uu['weight'][i])
                os.remove('./dummy/values_dummi_2_3_{0}.ini'.format(i))
                os.remove('./dummy/info_dummy_2_3_{0}.pkl'.format(i))
            resume['chi2_realization'] = chi2_realization
            resume['chi2_data'] = chi2_data
            resume['theory_dprime'] = theory_dprime
            resume['theory_d'] = theory_d
            
            resume['obs_dprime'] = obs_dprime
            resume['dprime_realization'] = dprime_realization
            resume['w'] = w
            resume['p_v'] = p_v
            resume['chi2_data0'] = chi2_data0
            
            save_obj('./dummy/dummy_2_3/resume_2_3_{0}'.format(index),resume)

    run_count = 0
    print (number_of_runs)
    
    while run_count<number_of_runs:
        print (run_count)
        comm = MPI.COMM_WORLD
        

        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if run_count+comm.rank<number_of_runs:
            doit(run_count+comm.rank,chunk_size,len_chain,uu)   
        
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
    # read all the files and append relevant quantitites
    resume = dict()
    resume['chi2_realization']  = []
    resume['chi2_data']= []
    resume['theory_dprime'] = []
    resume['obs_dprime'] = []
    resume['dprime_realization'] = []
    resume['w'] = []
    

    files = glob.glob('./dummy/dummy_2_3/resume_2_3_*')


    for file in files:
        m = load_obj(file.strip('.pkl'))
        for ii in range(len(m['chi2_realization'])):
            resume['chi2_realization'].append(m['chi2_realization'][ii] )
            resume['chi2_data'].append(m['chi2_data'][ii] )
            resume['theory_dprime'].append(m['theory_dprime'][ii] )
            resume['obs_dprime'].append(m['obs_dprime'][ii] )
            resume['dprime_realization'].append(m['dprime_realization'][ii] )
            resume['w'].append(m['w'][ii] )
            
    resume['chi2_realization'] = np.array(resume['chi2_realization'])
    resume['dprime_realization'] = np.array(resume['dprime_realization'])
    resume['w'] = np.array(resume['w'])
    resume['obs_dprime'] = np.array(resume['obs_dprime'])
    resume['theory_dprime'] = np.array(resume['theory_dprime'])
    resume['chi2_data'] = np.array(resume['chi2_data'])  
    save_obj('./dummy/resume_2_3',resume)




    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    import os
    # Read Unblinded 3rd chain ********* ********* ********* ********* ********* ********* ********* ********* ********* *********
    ou = '/global/cscratch1/sd/mgatti/Mass_Mapping/chains_cosmosis/chain_3tomocross_kEkE_UNBlinded.txt'
    uu = np.genfromtxt(ou, names = True)
    len_chain  = len(uu['cosmological_parametersomega_m'])
    chunk_size = 30
    number_of_runs = math.ceil(len_chain/chunk_size)
    if not os.path.exists('./dummy/dummy_3_2/'):
        os.mkdir('./dummy/dummy_3_2/')


    def doit(index,chunk_size,len_chain,uu): 
        ranges = [index*chunk_size,min([(index+1)*chunk_size,len_chain])]
        
        chi2_realization = []
        chi2_data = []
        chi2_data0 = []
        dprime_realization = []
        theory_dprime = []
        theory_d = []
        obs_dprime = []
        w =[]
        p_v = []
        resume = dict()
        if not os.path.exists('./dummy/dummy_3_2/Aresume_3_2_{0}.pkl'.format(index)):
            for i in range(ranges[0],ranges[1]):
                make_values(i,uu,'./dummy/values_dummi_3_2_{0}.ini'.format(i))
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_3_2.ini'
                os.environ["DATAFILE"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits'
                os.environ["DATAFILE_SR"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy'
                os.environ["SCALEFILE"] = 'Y3_params_priors/scales.ini'
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_3_2.ini'
                os.environ["VALUESINCLUDE"] = 'dummy/values_dummi_3_2_{0}.ini'.format(i)
                os.environ["PRIORSINCLUDE"] = 'Y3_params_priors/priors.ini'
                os.environ["output_info"] = 'info_dummy_3_2_{0}'.format(i)
                
         
                os.system('cosmosis Y3_params_priors/params.ini ')

                info_to_save =load_obj('./dummy/info_dummy_3_2_{0}'.format(i))
                chi2_realization.append(info_to_save['chi2_realization'])
                chi2_data.append(info_to_save['chi2_data'])
                dprime_realization.append(info_to_save['dprime_realization'])
                theory_dprime.append(info_to_save['theory_dprime'])
                theory_d.append(info_to_save['theory_d'])
                obs_dprime.append(info_to_save['obs_dprime'])
                p_v.append(info_to_save['p_v'])
                chi2_data0.append(info_to_save['chi2_data0'])
                w.append(uu['weight'][i])
                os.remove('./dummy/values_dummi_3_2_{0}.ini'.format(i))
                os.remove('./dummy/info_dummy_3_2_{0}.pkl'.format(i))
            resume['chi2_realization'] = chi2_realization
            resume['chi2_data'] = chi2_data

            resume['theory_dprime'] = theory_dprime
            resume['theory_d'] = theory_d
            resume['obs_dprime'] = obs_dprime
            resume['dprime_realization'] = dprime_realization
            resume['w'] = w
            resume['p_v'] = p_v
            resume['chi2_data0'] = chi2_data0
            
            save_obj('./dummy/dummy_3_2/resume_3_2_{0}'.format(index),resume)

    run_count = 0
    print (number_of_runs)
    
    while run_count<number_of_runs:
        print (run_count)
        comm = MPI.COMM_WORLD
        

        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if run_count+comm.rank<number_of_runs:
            doit(run_count+comm.rank,chunk_size,len_chain,uu)   
        
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
    # read all the files and append relevant quantitites
    resume = dict()
    resume['chi2_realization']  = []
    resume['chi2_data']= []
    resume['theory_dprime'] = []
    resume['obs_dprime'] = []
    resume['dprime_realization'] = []
    resume['w'] = []
    

    files = glob.glob('./dummy/dummy_3_2/resume_3_2_*')


    for file in files:
        m = load_obj(file.strip('.pkl'))
        for ii in range(len(m['chi2_realization'])):
            resume['chi2_realization'].append(m['chi2_realization'][ii] )
            resume['chi2_data'].append(m['chi2_data'][ii] )
            resume['theory_dprime'].append(m['theory_dprime'][ii] )
            resume['obs_dprime'].append(m['obs_dprime'][ii] )
            resume['dprime_realization'].append(m['dprime_realization'][ii] )
            resume['w'].append(m['w'][ii] )
            
    resume['chi2_realization'] = np.array(resume['chi2_realization'])
    resume['dprime_realization'] = np.array(resume['dprime_realization'])
    resume['w'] = np.array(resume['w'])
    resume['obs_dprime'] = np.array(resume['obs_dprime'])
    resume['theory_dprime'] = np.array(resume['theory_dprime'])
    resume['chi2_data'] = np.array(resume['chi2_data'])  
    save_obj('./dummy/resume_3_2',resume)
    '''
    
    
    
    
    
    '''   
    
    binx = 1
    
    
    
    
    import os
    
    # Read Unblinded 2nd chain ********* ********* ********* ********* ********* ********* ********* ********* ********* *********
    ou = '/global/cscratch1/sd/mgatti/Mass_Mapping/chains_cosmosis/chain_2_3tomocross_kEkE_UNBlinded_{0}binNO.txt'.format(binx)

    uu = np.genfromtxt(ou, names = True)
    len_chain  = len(uu['cosmological_parametersomega_m'])
    chunk_size = 30
    number_of_runs = math.ceil(len_chain/chunk_size)
    if not os.path.exists('./dummy/dummy_{0}no/'.format(binx)):
        os.mkdir('./dummy/dummy_{0}no/'.format(binx))


    def doit(index,chunk_size,len_chain,uu): 
        ranges = [index*chunk_size,min([(index+1)*chunk_size,len_chain])]
        
        chi2_realization = []
        chi2_data = []
        chi2_data0= []
        dprime_realization = []
        theory_dprime = []
        theory_d = []
        obs_dprime = []
        w =[]
        p_v = []
        
        resume = dict()
        if not os.path.exists('./dummy/dummy_{1}no/Aresume_{1}no_{0}.pkl'.format(index,binx)):
            for i in range(ranges[0],ranges[1]):
                make_values(i,uu,'./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_{0}no.ini'.format(binx)
                os.environ["DATAFILE"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits'
                os.environ["DATAFILE_SR"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy'
                os.environ["SCALEFILE"] = 'Y3_params_priors/scales.ini'
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_{0}no.ini'.format(binx)
                os.environ["VALUESINCLUDE"] = 'dummy/values_dummi_{1}no_{0}.ini'.format(i,binx)
                os.environ["PRIORSINCLUDE"] = 'Y3_params_priors/priors.ini'
                os.environ["output_info"] = 'info_dummy_{1}no_{0}'.format(i,binx)
                
         
                os.system('cosmosis Y3_params_priors/params.ini ')

                info_to_save =load_obj('./dummy/info_dummy_{1}no_{0}'.format(i,binx))
                chi2_realization.append(info_to_save['chi2_realization'])
                chi2_data.append(info_to_save['chi2_data'])
                dprime_realization.append(info_to_save['dprime_realization'])
                theory_dprime.append(info_to_save['theory_dprime'])
                theory_d.append(info_to_save['theory_d'])
                
                obs_dprime.append(info_to_save['obs_dprime'])
                p_v.append(info_to_save['p_v'])
                chi2_data0.append(info_to_save['chi2_data0'])
                w.append(uu['weight'][i])
                os.remove('./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.remove('./dummy/info_dummy_{1}no_{0}.pkl'.format(i,binx))
            resume['chi2_realization'] = chi2_realization
            resume['chi2_data'] = chi2_data
            resume['theory_dprime'] = theory_dprime
            resume['theory_d'] = theory_d
            
            resume['obs_dprime'] = obs_dprime
            resume['dprime_realization'] = dprime_realization
            resume['w'] = w
            resume['p_v'] = p_v
            resume['chi2_data0'] = chi2_data0
            
            save_obj('./dummy/dummy_{1}no/resume_{1}no_{0}'.format(index,binx),resume)

    run_count = 0
    print (number_of_runs)
    
    while run_count<number_of_runs:
        print (run_count)
        comm = MPI.COMM_WORLD
        

        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if run_count+comm.rank<number_of_runs:
            doit(run_count+comm.rank,chunk_size,len_chain,uu)   
        
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
    # read all the files and append relevant quantitites
    resume = dict()
    resume['chi2_realization']  = []
    resume['chi2_data']= []
    resume['theory_dprime'] = []
    resume['obs_dprime'] = []
    resume['dprime_realization'] = []
    resume['w'] = []
    

    files = glob.glob('./dummy/dummy_{0}no/resume_{0}no_*'.format(binx))


    for file in files:
        m = load_obj(file.strip('.pkl'))
        for ii in range(len(m['chi2_realization'])):
            resume['chi2_realization'].append(m['chi2_realization'][ii] )
            resume['chi2_data'].append(m['chi2_data'][ii] )
            resume['theory_dprime'].append(m['theory_dprime'][ii] )
            resume['obs_dprime'].append(m['obs_dprime'][ii] )
            resume['dprime_realization'].append(m['dprime_realization'][ii] )
            resume['w'].append(m['w'][ii] )
            
    resume['chi2_realization'] = np.array(resume['chi2_realization'])
    resume['dprime_realization'] = np.array(resume['dprime_realization'])
    resume['w'] = np.array(resume['w'])
    resume['obs_dprime'] = np.array(resume['obs_dprime'])
    resume['theory_dprime'] = np.array(resume['theory_dprime'])
    resume['chi2_data'] = np.array(resume['chi2_data'])  
    save_obj('./dummy/resume_{0}no'.format(binx),resume)




    
    import os
    binx = 'small'
    # Read Unblinded 2nd chain ********* ********* ********* ********* ********* ********* ********* ********* ********* *********
    ou = '/global/cscratch1/sd/mgatti/Mass_Mapping/chains_cosmosis/chain_2_3tomocross_kEkE_UNBlinded_SMALL2.txt'.format(binx)
    

    
    uu = np.genfromtxt(ou, names = True)
    len_chain  = len(uu['cosmological_parametersomega_m'])
    chunk_size = 30
    number_of_runs = math.ceil(len_chain/chunk_size)
    if not os.path.exists('./dummy/dummy_small/'.format(binx)):
        os.mkdir('./dummy/dummy_small/'.format(binx))


    def doit(index,chunk_size,len_chain,uu): 
        ranges = [index*chunk_size,min([(index+1)*chunk_size,len_chain])]
        
        chi2_realization = []
        chi2_data = []
        chi2_data0= []
        dprime_realization = []
        theory_dprime = []
        theory_d = []
        obs_dprime = []
        w =[]
        p_v = []
        
        resume = dict()
        if not os.path.exists('./dummy/dummy_small/Aresume_{1}no_{0}.pkl'.format(index,binx)):
            for i in range(ranges[0],ranges[1]):
                make_values(i,uu,'./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_small_large.ini'.format(binx)
                os.environ["DATAFILE"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits'
                os.environ["DATAFILE_SR"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy'
                os.environ["SCALEFILE"] = 'Y3_params_priors/scales.ini'
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_small_large.ini'.format(binx)
                os.environ["VALUESINCLUDE"] = 'dummy/values_dummi_{1}no_{0}.ini'.format(i,binx)
                os.environ["PRIORSINCLUDE"] = 'Y3_params_priors/priors.ini'
                os.environ["output_info"] = 'info_dummy_{1}no_{0}'.format(i,binx)
                
         
                os.system('cosmosis Y3_params_priors/params.ini ')

                info_to_save =load_obj('./dummy/info_dummy_{1}no_{0}'.format(i,binx))
                chi2_realization.append(info_to_save['chi2_realization'])
                chi2_data.append(info_to_save['chi2_data'])
                dprime_realization.append(info_to_save['dprime_realization'])
                theory_dprime.append(info_to_save['theory_dprime'])
                theory_d.append(info_to_save['theory_d'])
                
                obs_dprime.append(info_to_save['obs_dprime'])
                p_v.append(info_to_save['p_v'])
                chi2_data0.append(info_to_save['chi2_data0'])
                w.append(uu['weight'][i])
                os.remove('./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.remove('./dummy/info_dummy_{1}no_{0}.pkl'.format(i,binx))
            resume['chi2_realization'] = chi2_realization
            resume['chi2_data'] = chi2_data
            resume['theory_dprime'] = theory_dprime
            resume['theory_d'] = theory_d
            
            resume['obs_dprime'] = obs_dprime
            resume['dprime_realization'] = dprime_realization
            resume['w'] = w
            resume['p_v'] = p_v
            resume['chi2_data0'] = chi2_data0
            
            save_obj('./dummy/dummy_small/resume_{1}no_{0}'.format(index,binx),resume)
    run_count = 0
    print (number_of_runs)
    
    while run_count<number_of_runs:
        print (run_count)
        comm = MPI.COMM_WORLD
        

        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if run_count+comm.rank<number_of_runs:
            doit(run_count+comm.rank,chunk_size,len_chain,uu)   
        
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
    # read all the files and append relevant quantitites
    resume = dict()
    resume['chi2_realization']  = []
    resume['chi2_data']= []
    resume['theory_dprime'] = []
    resume['obs_dprime'] = []
    resume['dprime_realization'] = []
    resume['w'] = []
    

    files = glob.glob('./dummy/dummy_small/resume_{0}no_*'.format(binx))


    for file in files:
        m = load_obj(file.strip('.pkl'))
        for ii in range(len(m['chi2_realization'])):
            resume['chi2_realization'].append(m['chi2_realization'][ii] )
            resume['chi2_data'].append(m['chi2_data'][ii] )
            resume['theory_dprime'].append(m['theory_dprime'][ii] )
            resume['obs_dprime'].append(m['obs_dprime'][ii] )
            resume['dprime_realization'].append(m['dprime_realization'][ii] )
            resume['w'].append(m['w'][ii] )
            
    resume['chi2_realization'] = np.array(resume['chi2_realization'])
    resume['dprime_realization'] = np.array(resume['dprime_realization'])
    resume['w'] = np.array(resume['w'])
    resume['obs_dprime'] = np.array(resume['obs_dprime'])
    resume['theory_dprime'] = np.array(resume['theory_dprime'])
    resume['chi2_data'] = np.array(resume['chi2_data'])  
    save_obj('./dummy/resume_{0}no'.format(binx),resume)
 
    
    
    import os
    binx = 'large'
    # Read Unblinded 2nd chain ********* ********* ********* ********* ********* ********* ********* ********* ********* *********
    ou = '/global/cscratch1/sd/mgatti/Mass_Mapping/chains_cosmosis/chain_2_3tomocross_kEkE_UNBlinded_LARGE2.txt'.format(binx)
    

    
    uu = np.genfromtxt(ou, names = True)
    len_chain  = len(uu['cosmological_parametersomega_m'])
    chunk_size = 30
    number_of_runs = math.ceil(len_chain/chunk_size)
    if not os.path.exists('./dummy/dummy_large/'.format(binx)):
        os.mkdir('./dummy/dummy_large/'.format(binx))


    def doit(index,chunk_size,len_chain,uu): 
        ranges = [index*chunk_size,min([(index+1)*chunk_size,len_chain])]
        
        chi2_realization = []
        chi2_data = []
        chi2_data0= []
        dprime_realization = []
        theory_dprime = []
        theory_d = []
        obs_dprime = []
        w =[]
        p_v = []
        
        resume = dict()
        if not os.path.exists('./dummy/dummy_large/Aresume_{1}no_{0}.pkl'.format(index,binx)):
            for i in range(ranges[0],ranges[1]):
                make_values(i,uu,'./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_large_small.ini'.format(binx)
                os.environ["DATAFILE"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits'
                os.environ["DATAFILE_SR"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy'
                os.environ["SCALEFILE"] = 'Y3_params_priors/scales.ini'
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_large_small.ini'.format(binx)
                os.environ["VALUESINCLUDE"] = 'dummy/values_dummi_{1}no_{0}.ini'.format(i,binx)
                os.environ["PRIORSINCLUDE"] = 'Y3_params_priors/priors.ini'
                os.environ["output_info"] = 'info_dummy_{1}no_{0}'.format(i,binx)
                
         
                os.system('cosmosis Y3_params_priors/params.ini ')

                info_to_save =load_obj('./dummy/info_dummy_{1}no_{0}'.format(i,binx))
                chi2_realization.append(info_to_save['chi2_realization'])
                chi2_data.append(info_to_save['chi2_data'])
                dprime_realization.append(info_to_save['dprime_realization'])
                theory_dprime.append(info_to_save['theory_dprime'])
                theory_d.append(info_to_save['theory_d'])
                
                obs_dprime.append(info_to_save['obs_dprime'])
                p_v.append(info_to_save['p_v'])
                chi2_data0.append(info_to_save['chi2_data0'])
                w.append(uu['weight'][i])
                os.remove('./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.remove('./dummy/info_dummy_{1}no_{0}.pkl'.format(i,binx))
            resume['chi2_realization'] = chi2_realization
            resume['chi2_data'] = chi2_data
            resume['theory_dprime'] = theory_dprime
            resume['theory_d'] = theory_d
            
            resume['obs_dprime'] = obs_dprime
            resume['dprime_realization'] = dprime_realization
            resume['w'] = w
            resume['p_v'] = p_v
            resume['chi2_data0'] = chi2_data0
            
            save_obj('./dummy/dummy_large/resume_{1}no_{0}'.format(index,binx),resume)
    run_count = 0
    print (number_of_runs)
    
    while run_count<number_of_runs:
        print (run_count)
        comm = MPI.COMM_WORLD
        

        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if run_count+comm.rank<number_of_runs:
            doit(run_count+comm.rank,chunk_size,len_chain,uu)   
        
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
    # read all the files and append relevant quantitites
    resume = dict()
    resume['chi2_realization']  = []
    resume['chi2_data']= []
    resume['theory_dprime'] = []
    resume['obs_dprime'] = []
    resume['dprime_realization'] = []
    resume['w'] = []
    

    files = glob.glob('./dummy/dummy_large/resume_{0}no_*'.format(binx))


    for file in files:
        m = load_obj(file.strip('.pkl'))
        for ii in range(len(m['chi2_realization'])):
            resume['chi2_realization'].append(m['chi2_realization'][ii] )
            resume['chi2_data'].append(m['chi2_data'][ii] )
            resume['theory_dprime'].append(m['theory_dprime'][ii] )
            resume['obs_dprime'].append(m['obs_dprime'][ii] )
            resume['dprime_realization'].append(m['dprime_realization'][ii] )
            resume['w'].append(m['w'][ii] )
            
    resume['chi2_realization'] = np.array(resume['chi2_realization'])
    resume['dprime_realization'] = np.array(resume['dprime_realization'])
    resume['w'] = np.array(resume['w'])
    resume['obs_dprime'] = np.array(resume['obs_dprime'])
    resume['theory_dprime'] = np.array(resume['theory_dprime'])
    resume['chi2_data'] = np.array(resume['chi2_data'])  
    save_obj('./dummy/resume_{0}no'.format(binx),resume)
    '''
    
    
    
    

    
   
    import os
    binx = '3_auto'
    # Read Unblinded 2nd chain ********* ********* ********* ********* ********* ********* ********* ********* ********* *********
    ou = '/global/cscratch1/sd/mgatti/Mass_Mapping/chains_cosmosis/chain_3tomocross_kEkE_UNBlinded.txt'.format(binx)
    

    
    uu = np.genfromtxt(ou, names = True)
    len_chain  = len(uu['cosmological_parametersomega_m'])
    chunk_size = 30
    number_of_runs = math.ceil(len_chain/chunk_size)
    if not os.path.exists('./dummy/dummy_3_auto/'.format(binx)):
        os.mkdir('./dummy/dummy_3_auto/'.format(binx))


    def doit(index,chunk_size,len_chain,uu): 
        ranges = [index*chunk_size,min([(index+1)*chunk_size,len_chain])]
        
        chi2_realization = []
        chi2_data = []
        chi2_data0= []
        dprime_realization = []
        theory_dprime = []
        theory_d = []
        obs_dprime = []
        w =[]
        p_v = []
        
        resume = dict()
        if not os.path.exists('./dummy/dummy_3_auto/resume_{1}no_{0}.pkl'.format(index,binx)):
            for i in range(ranges[0],ranges[1]):
                make_values(i,uu,'./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_large_small.ini'.format(binx)
                os.environ["DATAFILE"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits'
                os.environ["DATAFILE_SR"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy'
                os.environ["SCALEFILE"] = 'Y3_params_priors/scales.ini'
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_3_auto.ini'.format(binx)
                os.environ["VALUESINCLUDE"] = 'dummy/values_dummi_{1}no_{0}.ini'.format(i,binx)
                os.environ["PRIORSINCLUDE"] = 'Y3_params_priors/priors.ini'
                os.environ["output_info"] = 'info_dummy_{1}no_{0}'.format(i,binx)
                
         
                os.system('cosmosis Y3_params_priors/params.ini ')

                info_to_save =load_obj('./dummy/info_dummy_{1}no_{0}'.format(i,binx))

                p_v.append(info_to_save['p_v'])
                w.append(uu['weight'][i])
                os.remove('./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.remove('./dummy/info_dummy_{1}no_{0}.pkl'.format(i,binx))

            resume['w'] = w
            resume['p_v'] = p_v
           
            save_obj('./dummy/dummy_3_auto/resume_{1}no_{0}'.format(index,binx),resume)
    run_count = 0
    print (number_of_runs)
    
    while run_count<number_of_runs:
        print (run_count)
        comm = MPI.COMM_WORLD
        

        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if run_count+comm.rank<number_of_runs:
            doit(run_count+comm.rank,chunk_size,len_chain,uu)   
        
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        

        

        
        
        
        
        
        
        
        
    '''
        
        
    import os
    binx = '23_auto_SR'
    # Read Unblinded 2nd chain ********* ********* ********* ********* ********* ********* ********* ********* ********* *********
    ou = '/global/cscratch1/sd/mgatti/Mass_Mapping/chains_cosmosis/chain_2_3tomocross_kEkE_UNBlinded.txt'.format(binx)
    

    
    uu = np.genfromtxt(ou, names = True)
    len_chain  = len(uu['cosmological_parametersomega_m'])
    chunk_size = 30
    number_of_runs = math.ceil(len_chain/chunk_size)
    if not os.path.exists('./dummy/dummy_23_auto_SR/'.format(binx)):
        os.mkdir('./dummy/dummy_23_auto_SR/'.format(binx))


    def doit(index,chunk_size,len_chain,uu): 
        ranges = [index*chunk_size,min([(index+1)*chunk_size,len_chain])]
        
        chi2_realization = []
        chi2_data = []
        chi2_data0= []
        dprime_realization = []
        theory_dprime = []
        theory_d = []
        obs_dprime = []
        w =[]
        p_v = []
        
        resume = dict()
        if not os.path.exists('./dummy/dummy_23_auto_SR/resume_{1}no_{0}.pkl'.format(index,binx)):
            for i in range(ranges[0],ranges[1]):
                make_values(i,uu,'./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.environ["DATAFILE"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits'
                os.environ["DATAFILE_SR"] = 'data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy'
                os.environ["SCALEFILE"] = 'Y3_params_priors/scales.ini'
                os.environ["INCLUDEFILE"] = 'Y3_params_priors/PPD_23_auto_SR.ini'.format(binx)
                os.environ["VALUESINCLUDE"] = 'dummy/values_dummi_{1}no_{0}.ini'.format(i,binx)
                os.environ["PRIORSINCLUDE"] = 'Y3_params_priors/priors.ini'
                os.environ["output_info"] = 'info_dummy_{1}no_{0}'.format(i,binx)
                
         
                os.system('cosmosis Y3_params_priors/params.ini ')

                info_to_save =load_obj('./dummy/info_dummy_{1}no_{0}'.format(i,binx))

                p_v.append(info_to_save['p_v'])
                w.append(uu['weight'][i])
                os.remove('./dummy/values_dummi_{1}no_{0}.ini'.format(i,binx))
                os.remove('./dummy/info_dummy_{1}no_{0}.pkl'.format(i,binx))

            resume['w'] = w
            resume['p_v'] = p_v
           
            save_obj('./dummy/dummy_23_auto_SR/resume_{1}no_{0}'.format(index,binx),resume)
    run_count = 0
    print (number_of_runs)
    
    while run_count<number_of_runs:
        print (run_count)
        comm = MPI.COMM_WORLD
        

        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if run_count+comm.rank<number_of_runs:
            doit(run_count+comm.rank,chunk_size,len_chain,uu)   
        
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
    
    '''
    