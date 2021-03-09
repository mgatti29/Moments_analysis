from Moments_analysis import moments_map
import pickle 
import healpy as hp
import numpy as np
import os
from astropy.table import Table
import gc
from mpi4py import MPI 
import pyfits as pf

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f)
        f.close()
    return mute

'''
The code computes moments from density /shear maps generated with FLASK, for the purpose of generating a covariance matrix.
You need to run first the code 'convert_flask2maps.py'. As this code uses a lot of memory, the first time you run it it will
only smooth the maps. It's divided such that you can run multiple jobs - each of them will take a few hours. It will save the 
maps to disk, so if it's not done, you can keep running the code until all the maps are saved. Once that is done, you can set 
compute_moments = True. In such mode, the code will compute the moments of the smoothed
maps. The code in this mode will use more memory, but it only take ~ 15 minutes per simulation.

The first part can be run using ()
srun --nodes=20 --tasks-per-node=15 --cpus-per-task=4 --cpu-bind=cores --mem=110GB python run_flask_measurements_2x2.py

As for the second:
srun --nodes=20 --tasks-per-node=2 --cpus-per-task=30 --cpu-bind=cores --mem=110GB python run_flask_measurements_2x2.py

'''


rewrite = False
compute_moments = True#True
output_FLASK = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/flask_tests/'
output_folder = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/flask_tests/'
FLASK_path = '/global/cscratch1/sd/faoli/flask_desy3/4096/'

n_FLASK_real = 100
tomo_bins = [0,1,2,3]
tomo_bins_lens = [0,1,2,3,4]


fields_to_compute = {'original_k_full' : True, #this is not sampled by galaxies --
 'original_k' : True, #this is not sampled by galaxies --
 'noisy_k' : True,   
 'noise_k' : True,  
 'noiseless_k' : True,  
 'galaxy_d' : True, 
 'density_d' : True # this is not sampled by galaxies --
}  

def runit(seed, chunk):

    conf = dict()
    conf['smoothing_scales'] = np.array([3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.]) # arcmin
    conf['nside'] = 1024
    conf['lmax'] = 2048
    conf['verbose'] = True
    conf['output_folder'] = output_folder + '/output_'+str(seed+1)+'/'
    
    if not os.path.exists(conf['output_folder']):
        os.mkdir(conf['output_folder'])

    
    # to allow for parallel execution of the smoothing part, we load and smooth the maps in different chunks and perform the moment computation in a second run.
    if chunk == 'density_d':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            mcal_moments = moments_map(conf)
            
            for i in range(len(tomo_bins_lens)):
                mcal_moments.add_map(FLASK_catalog['density_full'][i], field_label = 'density_full', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['density'][i], field_label = 'density', tomo_bin = i)
            del FLASK_catalog 
            gc.collect()
            mcal_moments.transform_and_smooth('density_full','density_full', shear = False, tomo_bins = tomo_bins_lens, overwrite = False, skip_loading_smoothed_maps = True)   
            mcal_moments.transform_and_smooth('density','density', shear = False, tomo_bins = tomo_bins_lens, overwrite = False, skip_loading_smoothed_maps = True)  
            del mcal_moments
            gc.collect()
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1),['Done'])
    

            
            
    if chunk == 'original_k':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            
            # first run for shear *********************
            # load FLASK maps *******************************
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            # initialise obj *********
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in range(len(tomo_bins)):

                mcal_moments.add_map(FLASK_catalog['sources'][i]['g1_field'], field_label = 'g1_field', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['sources'][i]['g2_field'], field_label = 'g2_field', tomo_bin = i)
                
            del FLASK_catalog 
            gc.collect()   
            mcal_moments.transform_and_smooth('convergence_field','g1_field','g2_field', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)     
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1),['Done'])
            del mcal_moments
            gc.collect()
            
            
    if chunk == 'original_k_full':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            
            # first run for shear *********************
            # load FLASK maps *******************************
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            # initialise obj *********
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in range(len(tomo_bins)):

                mcal_moments.add_map(FLASK_catalog['sources'][i]['g1_field_full'], field_label = 'g1_field_full', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['sources'][i]['g2_field_full'], field_label = 'g2_field_full', tomo_bin = i)
                
            del FLASK_catalog 
            gc.collect()   
            mcal_moments.transform_and_smooth('convergence_field_full','g1_field_full','g2_field_full', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)     
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1),['Done'])
            del mcal_moments
            gc.collect()
            
            
    if chunk == 'noiseless_k':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            
            # first run for shear *********************
            # load FLASK maps *******************************
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            # initialise obj *********
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in range(len(tomo_bins)):

                mcal_moments.add_map(FLASK_catalog['sources'][i]['g1_map'], field_label = 'g1', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['sources'][i]['g2_map'], field_label = 'g2', tomo_bin = i)
                
            del FLASK_catalog 
            gc.collect()   
            mcal_moments.transform_and_smooth('convergence_noiseless','g1','g2', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)     
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1),['Done'])
            del mcal_moments
            gc.collect()
            
            
            
            
    if chunk == 'noisy_k':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            
            # first run for shear *********************
            # load FLASK maps *******************************
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            # initialise obj *********
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in range(len(tomo_bins)):
                mcal_moments.add_map(FLASK_catalog['sources'][i]['e1_map'], field_label = 'e1', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['sources'][i]['e2_map'], field_label = 'e2', tomo_bin = i)

            del FLASK_catalog 
            gc.collect()
            mcal_moments.transform_and_smooth('convergence','e1','e2', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)         
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1),['Done'])
            del mcal_moments
            gc.collect()
            
    if chunk == 'noise_k':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            # second run for shape noise *********************
            # load FLASK maps *******************************
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in range(len(tomo_bins)):
                mcal_moments.add_map(FLASK_catalog['sources'][i]['e1r_map'], field_label = 'e1r', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['sources'][i]['e2r_map'], field_label = 'e2r', tomo_bin = i)
            del FLASK_catalog
            gc.collect()
            mcal_moments.transform_and_smooth('noise','e1r','e2r', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)   
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1),['Done'])
        
            del mcal_moments
            gc.collect()
            
    if chunk == 'galaxy_d':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            mcal_moments = moments_map(conf)
            
            for i in range(len(tomo_bins_lens)):
                mcal_moments.add_map(FLASK_catalog['lenses'][i], field_label = 'g', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['randoms'][i], field_label = 'r', tomo_bin = i)
            del FLASK_catalog 
            gc.collect()
            mcal_moments.transform_and_smooth('galaxies','g', shear = False, tomo_bins = tomo_bins_lens, overwrite = False, skip_loading_smoothed_maps = True)   
            mcal_moments.transform_and_smooth('randoms','r', shear = False, tomo_bins = tomo_bins_lens, overwrite = False, skip_loading_smoothed_maps = True)  
            del mcal_moments
            gc.collect()
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed_'+str(seed+1),['Done'])
    
    if compute_moments:
        print ('computing moments')
        mcal_moments = moments_map(conf)
        

        mcal_moments.transform_and_smooth('convergence_field_full','g1_field_full','g2_field_full', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True)      

        mcal_moments.transform_and_smooth('convergence_field','g1_field','g2_field', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True)   

        mcal_moments.transform_and_smooth('density_full','density_full', shear = False, tomo_bins = tomo_bins_lens, overwrite = False, skip_conversion_toalm = True)   
        mcal_moments.transform_and_smooth('density','density', shear = False, tomo_bins = tomo_bins_lens, overwrite = False, skip_conversion_toalm = True)   
            
            
        mcal_moments.transform_and_smooth('convergence_noiseless','g1','g2', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True)    
        mcal_moments.transform_and_smooth('convergence','e1','e2', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True)      
        mcal_moments.transform_and_smooth('noise','e1r','e2r', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True) 
        mcal_moments.transform_and_smooth('galaxies','g', shear = False, tomo_bins = tomo_bins_lens, overwrite = False , skip_conversion_toalm = True)    
        mcal_moments.transform_and_smooth('randoms','r', shear = False, tomo_bins = tomo_bins_lens, overwrite = False , skip_conversion_toalm = True)                 
    
            
        del mcal_moments.fields 
        gc.collect()
    
        # We don't de-noise here - we do it at posteriori when assemplying the covariance.
        mcal_moments.compute_moments( label_moments='field_full_kEkE', field_label1 ='convergence_field_full_kE',  tomo_bins1 = tomo_bins)
        mcal_moments.compute_moments( label_moments='field_kEkE', field_label1 ='convergence_field_kE',  tomo_bins1 = tomo_bins_lens)
        mcal_moments.compute_moments( label_moments='density_full_kEkE', field_label1 ='density_full_kE',  tomo_bins1 = tomo_bins_lens)
        mcal_moments.compute_moments( label_moments='density_kEkE', field_label1 ='density_kE',  tomo_bins1 = tomo_bins)
        
        mcal_moments.compute_moments( label_moments='density_KK_full', field_label1 ='density_full_kE', field_label2 = 'convergence_field_full_kE',  tomo_bins1 = tomo_bins_lens, tomo_bins2 = tomo_bins)
        mcal_moments.compute_moments( label_moments='K_densitydensity_full', field_label2 ='density_full_kE', field_label1 = 'convergence_field_full_kE',   tomo_bins2 = tomo_bins_lens, tomo_bins1 = tomo_bins)
        mcal_moments.compute_moments( label_moments='density_KK', field_label1 ='density_kE', field_label2 = 'convergence_field_kE',  tomo_bins1 = tomo_bins_lens, tomo_bins2 = tomo_bins)
        mcal_moments.compute_moments( label_moments='K_densitydensity', field_label2 ='density_kE', field_label1 = 'convergence_field_kE',   tomo_bins2 = tomo_bins_lens, tomo_bins1 = tomo_bins)
        
        
        
        
        
        mcal_moments.compute_moments( label_moments='noiseless_kEkE', field_label1 ='convergence_noiseless_kE',  tomo_bins1 = tomo_bins)
        mcal_moments.compute_moments( label_moments='noiseless_kBkB', field_label1 ='convergence_noiseless_kB',  tomo_bins1 = tomo_bins)

        mcal_moments.compute_moments( label_moments='kEkE', field_label1 ='convergence_kE',  tomo_bins1 = tomo_bins)
        mcal_moments.compute_moments( label_moments='kBkB', field_label1 ='convergence_kB',  tomo_bins1 = tomo_bins)
    
        # k NN needs to be subtracted for corrections to the third moments ***
        mcal_moments.compute_moments( label_moments='kEkN', field_label1 ='convergence_kE', field_label2 = 'noise_kE', tomo_bins1 = tomo_bins)
        mcal_moments.compute_moments( label_moments='kNkN', field_label1 ='noise_kE', field_label2 = 'noise_kE', tomo_bins1 = tomo_bins)
        mcal_moments.compute_moments( label_moments='kNBkNB', field_label1 ='noise_kB', field_label2 = 'noise_kB', tomo_bins1 = tomo_bins)
        mcal_moments.compute_moments( label_moments='kNkE', field_label2 ='convergence_kE', field_label1 = 'noise_kE',  tomo_bins1 = tomo_bins)
    
        mcal_moments.compute_moments( label_moments='dKK', field_label1 ='galaxies_kE', field_label2 = 'convergence_kE',  tomo_bins1 = tomo_bins_lens, tomo_bins2 = tomo_bins)
        mcal_moments.compute_moments( label_moments='Kdd', field_label2 ='galaxies_kE', field_label1 = 'convergence_kE',   tomo_bins2 = tomo_bins_lens, tomo_bins1 = tomo_bins)
        mcal_moments.compute_moments( label_moments='dkNkN', field_label2 ='noise_kE', field_label1 = 'density_kE',  tomo_bins1 = tomo_bins_lens, tomo_bins2 = tomo_bins)
        mcal_moments.compute_moments( label_moments='nKK', field_label2 ='convergence_kE', field_label1 = 'randoms_kE',  tomo_bins1 = tomo_bins_lens, tomo_bins2 = tomo_bins)
        mcal_moments.compute_moments( label_moments='nkNkN', field_label2 ='noise_kE', field_label1 = 'randoms_kE',  tomo_bins1 = tomo_bins_lens, tomo_bins2 = tomo_bins)

        mcal_moments.compute_moments( label_moments='kNdd', field_label1 ='noise_kE', field_label2 = 'density_kE',  tomo_bins1 = tomo_bins, tomo_bins2 = tomo_bins_lens)


        mcal_moments.compute_moments( label_moments='knn', field_label1 ='convergence_kE', field_label2 = 'randoms_kE',  tomo_bins1 = tomo_bins, tomo_bins2 = tomo_bins_lens)
        mcal_moments.compute_moments( label_moments='kNnn', field_label1 ='noise_kE', field_label2 = 'randoms_kE',  tomo_bins1 = tomo_bins, tomo_bins2 = tomo_bins_lens)
        mcal_moments.compute_moments( label_moments='dd', field_label2 ='galaxies_kE', field_label1 = 'galaxies_kE',  tomo_bins1 = tomo_bins_lens)
        mcal_moments.compute_moments( label_moments='nn', field_label2 ='randoms_kE', field_label1 = 'randoms_kE',  tomo_bins1 = tomo_bins_lens)
        mcal_moments.compute_moments( label_moments='dnn', field_label1 ='galaxies_kE', field_label2 = 'randoms_kE',  tomo_bins1 = tomo_bins_lens)
        mcal_moments.compute_moments( label_moments='ndd', field_label1 ='randoms_kE', field_label2 = 'density_kE',  tomo_bins1 = tomo_bins_lens)

        # I want to compute the full power spectrum of FLASK covariance for KS and density field as well. 
        PS_dict = dict()
        for i in tomo_bins:
            for j in tomo_bins:
                try:
                    mute = np.loadtxt(FLASK_path+'seed'+str(seed+1)+'/info_'+str(seed+1)+'/regCl-f10z{0}f10z{1}.dat'.format(i+1,j+1))
                except:
                    mute = np.loadtxt(FLASK_path+'seed'+str(seed+1)+'/regCl-f10z{0}f10z{1}.dat'.format(i+1,j+1))
                            
                PS_dict['k_'+str(i+1)+'_'+str(j+1)] = mute[:,1]
                del mute
            
        for i in tomo_bins_lens:
            for j in tomo_bins_lens:
                try:
                    mute = np.loadtxt(FLASK_path+'seed'+str(seed+1)+'/info_'+str(seed+1)+'/regCl-f{0}z{0}f{1}z{1}.dat'.format(i+1,j+1))
                except:
                    mute = np.loadtxt(FLASK_path+'seed'+str(seed+1)+'/regCl-f{0}z{0}f{1}z{1}.dat'.format(i+1,j+1))
                            
                PS_dict['lens_'+str(i+1)+'_'+str(j+1)] = mute[:,1]
                del mute
        
                                  
        for i in tomo_bins_lens:
            for j in tomo_bins:
                try:
                    mute = np.loadtxt(FLASK_path+'seed'+str(seed+1)+'/info_'+str(seed+1)+'/regCl-f{0}z{0}f10z{1}.dat'.format(i+1,j+1))
                except:
                    mute = np.loadtxt(FLASK_path+'seed'+str(seed+1)+'/regCl-f{0}z{0}f10z{1}.dat'.format(i+1,j+1))
                    
                            
                PS_dict['k_lens_'+str(i+1)+'_'+str(j+1)] = mute[:,1]
                del mute
                                  
                                  
        mcal_moments.PS = PS_dict
        
        try:
            del mcal_moments.smoothed_maps
            gc.collect()
        except:
            pass
        
        save_obj(output_folder+'moments_seed_'+str(seed+1),mcal_moments)
        # delete the folder.
        #os.system('rm -r '+conf['output_folder'])
if __name__ == '__main__':
     
    if not os.path.exists(output_folder+'/counts/'):
        os.mkdir(output_folder+'/counts/')              
    runstodo = []
    chunks_to_do =[]
                                  
      
                              
    for i in range(0,n_FLASK_real):
        for key in fields_to_compute.keys():
            if fields_to_compute[key]:        
                if not os.path.exists(output_folder+'/counts/'+str(key)+'_seed_'+str(i+1)+'.pkl'):
                    runstodo.append(i)
                    chunks_to_do.append(key)
            
        if not os.path.exists(output_folder+'moments_seed1A_'+str(i+1)+'.pkl'):
            if compute_moments:
                runstodo.append(i)
                chunks_to_do.append(-1)

    run_count=0
    while run_count<len(runstodo):
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        try:
            runit(runstodo[run_count+comm.rank],chunks_to_do[run_count+comm.rank])
        except:
            pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
