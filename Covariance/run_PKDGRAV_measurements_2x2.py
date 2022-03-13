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

The first part can be run setting 'compute_moments = False' and:
srun --nodes=1 --tasks-per-node=12 --cpus-per-task=5 --cpu-bind=cores --mem=110GB python run_PKDGRAV_measurements_2x2.py

As for the second, once the first part has completed all the runs, you need to set 'compute_moments = True' and run:
srun --nodes=5 --tasks-per-node=3 --cpus-per-task=20 --cpu-bind=cores --mem=110GB python run_PKDGRAV_measurements_2x2.py

'''


# this must be the same label you used in the convert_PKDGRAV_2maps
extralab_ ='_fid'


rewrite = False
compute_moments = False

output_folder = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/PKDGRAV_tests/'

# number of realisations you want to run
n_real = 12

# number of tomographic bins you're interested into. Just run 1 of them for the moment
tomo_bins = [2]



# you want to run 'noise_only' and 'noisy_k'.
fields_to_compute = { 'noisy_k' : True,   
 'noise_only' : True,  
 'noiseless_k' : False,  
}  

def runit(seed, chunk):

    conf = dict()
    conf['smoothing_scales'] = np.array([3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.]) # arcmin
    conf['nside'] = 1024
    conf['lmax'] = 2048
    conf['verbose'] = True
    conf['output_folder'] = output_folder + '/output'+extralab_+'_'+str(seed+1)+'/'
    
    if not os.path.exists(conf['output_folder']):
        try:
            os.mkdir(conf['output_folder'])
        except:
            pass
    
    # to allow for parallel execution of the smoothing part, we load and smooth the maps in different chunks and perform the moment computation in a second run.

            
            

            
            
    if chunk == 'noiseless_k':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed'+extralab_+'_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            
            # first run for shear *********************
            # load maps *******************************
            catalog = load_obj(output_folder+'seed'+extralab_+'_'+str(seed+1))
    
            # initialise obj *********
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in tomo_bins:

                mcal_moments.add_map(catalog['sources'][i]['g1_map'], field_label = 'g1', tomo_bin = i)
                mcal_moments.add_map(catalog['sources'][i]['g2_map'], field_label = 'g2', tomo_bin = i)
                
            del catalog 
            gc.collect()   
            mcal_moments.transform_and_smooth('convergence_noiseless','g1','g2', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)     
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed'+extralab_+'_'+str(seed+1),['Done'])
            del mcal_moments
            gc.collect()
            
            
            
            
    if chunk == 'noisy_k':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed'+extralab_+'_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            
            # first run for shear *********************
            # load FLASK maps *******************************
            catalog = load_obj(output_folder+'seed'+extralab_+'_'+str(seed+1))
    
            # initialise obj *********
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in tomo_bins:
                mcal_moments.add_map(catalog['sources'][i]['e1_map'], field_label = 'e1', tomo_bin = i)
                mcal_moments.add_map(catalog['sources'][i]['e2_map'], field_label = 'e2', tomo_bin = i)

            del catalog 
            gc.collect()
            mcal_moments.transform_and_smooth('convergence','e1','e2', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)         
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed'+extralab_+'_'+str(seed+1),['Done'])
            del mcal_moments
            gc.collect()
            
    if chunk == 'noise_only':
        path_check = output_folder+'/counts/'+str(chunk)+'_seed'+extralab_+'_'+str(seed+1)+'.pkl'
        if (not os.path.exists(path_check) or rewrite):
            # second run for shape noise *********************
            # load FLASK maps *******************************
            catalog = load_obj(output+'seed'+extralab_+'_'+str(seed+1))
    
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in tomo_bins:
                mcal_moments.add_map(catalog['sources'][i]['e1r_map'], field_label = 'e1r', tomo_bin = i)
                mcal_moments.add_map(catalog['sources'][i]['e2r_map'], field_label = 'e2r', tomo_bin = i)
            del catalog
            gc.collect()
            mcal_moments.transform_and_smooth('noise','e1r','e2r', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)   
            save_obj(output_folder+'/counts/'+str(chunk)+'_seed'+extralab_+'_'+str(seed+1),['Done'])
        
            del mcal_moments
            gc.collect()
            

    
    
    
    
    
    if compute_moments:
        print ('computing moments')
        mcal_moments = moments_map(conf)
        

        

        if fields_to_compute['original_k_full']:
            mcal_moments.transform_and_smooth('convergence_field_full','g1_field_full','g2_field_full', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True)      
        if fields_to_compute['original_k']:
            mcal_moments.transform_and_smooth('convergence_field','g1_field','g2_field', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True)   

        if fields_to_compute['density_d']:
            mcal_moments.transform_and_smooth('density_full','density_full', shear = False, tomo_bins = tomo_bins_lens, overwrite = False, skip_conversion_toalm = True)   
            mcal_moments.transform_and_smooth('density','density', shear = False, tomo_bins = tomo_bins_lens, overwrite = False, skip_conversion_toalm = True)   
            
        if fields_to_compute['noiseless_k']:
            mcal_moments.transform_and_smooth('convergence_noiseless','g1','g2', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True)    
        if fields_to_compute['noisy_k']:
            mcal_moments.transform_and_smooth('convergence','e1','e2', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True)      
        if fields_to_compute['noisy_k']:
            mcal_moments.transform_and_smooth('noise','e1r','e2r', shear = True, tomo_bins = tomo_bins, overwrite = False , skip_conversion_toalm = True) 
        if fields_to_compute['galaxy_d']:
            mcal_moments.transform_and_smooth('galaxies','g', shear = False, tomo_bins = tomo_bins_lens, overwrite = False , skip_conversion_toalm = True)    
            mcal_moments.transform_and_smooth('randoms','r', shear = False, tomo_bins = tomo_bins_lens, overwrite = False , skip_conversion_toalm = True)                 
    
            
        del mcal_moments.fields 
        gc.collect()
        
        # We don't de-noise here - we do it at posteriori when assemplying the covariance.
        if fields_to_compute['original_k_full']:
            mcal_moments.compute_moments( label_moments='field_full_kEkE', field_label1 ='convergence_field_full_kE',  tomo_bins1 = tomo_bins)
        if fields_to_compute['original_k']:
            mcal_moments.compute_moments( label_moments='field_kEkE', field_label1 ='convergence_field_kE',  tomo_bins1 = tomo_bins)
        if fields_to_compute['noiseless_k']:
            mcal_moments.compute_moments( label_moments='noiseless_kEkE', field_label1 ='convergence_noiseless_kE',  tomo_bins1 = tomo_bins)
            mcal_moments.compute_moments( label_moments='noiseless_kBkB', field_label1 ='convergence_noiseless_kB',  tomo_bins1 = tomo_bins)
        if fields_to_compute['noisy_k']:
            mcal_moments.compute_moments( label_moments='kEkE', field_label1 ='convergence_kE',  tomo_bins1 = tomo_bins)
            mcal_moments.compute_moments( label_moments='kBkB', field_label1 ='convergence_kB',  tomo_bins1 = tomo_bins)
            mcal_moments.compute_moments( label_moments='kEkN', field_label1 ='convergence_kE', field_label2 = 'noise_kE', tomo_bins1 = tomo_bins)
            mcal_moments.compute_moments( label_moments='kNkN', field_label1 ='noise_kE', field_label2 = 'noise_kE', tomo_bins1 = tomo_bins)
            mcal_moments.compute_moments( label_moments='kNBkNB', field_label1 ='noise_kB', field_label2 = 'noise_kB', tomo_bins1 = tomo_bins)
            mcal_moments.compute_moments( label_moments='kNkE', field_label2 ='convergence_kE', field_label1 = 'noise_kE',  tomo_bins1 = tomo_bins)
            

        save_obj(output_folder+'moments_seed'+extralab_+'_'+str(seed+1),mcal_moments)

if __name__ == '__main__':
     
    try:
        if not os.path.exists(output_folder+'/counts/'):
            os.mkdir(output_folder+'/counts/')     
    except:
        pass            
    runstodo = []
    chunks_to_do =[]
                                  
      
    # it checks what to run:
    for i in range(0,n_real):
        if not os.path.exists(output_folder+'moments_seedu'+extralab_+'_'+str(i+1)+'.pkl'):
            for key in fields_to_compute.keys():
                if fields_to_compute[key]:        
                    if not compute_moments:
                        if not os.path.exists(output_folder+'/counts/'+str(key)+'_seed'+extralab_+'_'+str(i+1)+'.pkl'):
                            runstodo.append(i)
                            chunks_to_do.append(key)
                
            if not os.path.exists(output_folder+'moments_seed'+extralab_+'_'+str(i+1)+'.pkl'):
                pass_ = True
                for key in fields_to_compute.keys():
                    if fields_to_compute[key]:        
                        if not os.path.exists(output_folder+'/counts/'+str(key)+'_seed'+extralab_+'_'+str(i+1)+'.pkl'):
                            pass_ = False
                if pass_:                              
                    if compute_moments:
                        runstodo.append(i)
                        chunks_to_do.append(-1)
#
    run_count=0
    print ('runs to do: ',len(runstodo))
    print (runstodo)
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
      # runit(runstodo[run_count],chunks_to_do[run_count])
      # run_count+=1

        
