from Moments_analysis import moments_map
import pickle 
import healpy as hp
import numpy as np
import os
from astropy.table import Table
import gc
from mpi4py import MPI 
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
srun --nodes=30 --tasks-per-node=20 --cpus-per-task=3 --cpu-bind=cores --mem=110GB python run_flask_measurements_2x2.py

As for the second:
srun --nodes=30 --tasks-per-node=3 --cpus-per-task=20 --cpu-bind=cores --mem=110GB python run_flask_measurements_2x2.py

'''

compute_moments = True
output_FLASK = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/flask/'

n_FLASK_real = 600

def runit(seed, chunk):
    
    conf = dict()
    conf['smoothing_scales'] = np.array([3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.]) # arcmin
    conf['nside'] = 1024
    conf['lmax'] = 2048
    conf['verbose'] = True
    conf['output_folder'] = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/flask/output_'+str(seed+1)+'/'
    if not os.path.exists(conf['output_folder']):
        os.mkdir(conf['output_folder'])

    
    # to allow for parallel execution of the smoothing part, we load and smooth the maps in three chunks and perform the moment computation in a second run.
    if chunk == 1:
        path_check = output_FLASK+'chunk_1_smoothed_seed_'+str(seed+1)+'.pkl'
        if not os.path.exists(path_check):
            
            # first run for shear *********************
            # load FLASK maps *******************************
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            # initialise obj *********
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in range(4):
                mcal_moments.add_map(FLASK_catalog['sources'][i]['e1_map'], field_label = 'e1', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['sources'][i]['e2_map'], field_label = 'e2', tomo_bin = i)
            del FLASK_catalog 
            gc.collect()
            mcal_moments.transform_and_smooth('convergence','e1','e2', shear = True, tomo_bins = [0,1,2,3], overwrite = False , skip_loading_smoothed_maps = True)     
            save_obj(output_FLASK+'chunk_1_smoothed_seed_'+str(seed+1),['Done'])
            del mcal_moments
            gc.collect()
            
    if chunk == 2:
        path_check = output_FLASK+'chunk_2_smoothed_seed_'+str(seed+1)+'.pkl'
        if not os.path.exists(path_check):
            # second run for shape noise *********************
            # load FLASK maps *******************************
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            mcal_moments = moments_map(conf)
            
            # assign maps *************
            for i in range(4):
                mcal_moments.add_map(FLASK_catalog['sources'][i]['e1r_map'], field_label = 'e1r', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['sources'][i]['e2r_map'], field_label = 'e2r', tomo_bin = i)
            del FLASK_catalog
            gc.collect()
            mcal_moments.transform_and_smooth('noise','e1r','e2r', shear = True, tomo_bins = [0,1,2,3], overwrite = False , skip_loading_smoothed_maps = True)   
            save_obj(output_FLASK+'chunk_2_smoothed_seed_'+str(seed+1),['Done'])
        
            del mcal_moments
            gc.collect()
            
    if chunk == 3:
        path_check = output_FLASK+'chunk_3_smoothed_seed_'+str(seed+1)+'.pkl'
        if not os.path.exists(path_check):
            FLASK_catalog = load_obj(output_FLASK+'seed_'+str(seed+1))
    
            mcal_moments = moments_map(conf)
            
            for i in range(5):
                mcal_moments.add_map(FLASK_catalog['lenses'][i], field_label = 'd', tomo_bin = i)
                mcal_moments.add_map(FLASK_catalog['randoms'][i], field_label = 'r', tomo_bin = i)
            del FLASK_catalog 
            gc.collect()
            mcal_moments.transform_and_smooth('density','d', shear = False, tomo_bins = [0,1,2,3,4], overwrite = False, skip_loading_smoothed_maps = True)   
            mcal_moments.transform_and_smooth('randoms','r', shear = False, tomo_bins = [0,1,2,3,4], overwrite = False, skip_loading_smoothed_maps = True)  
            del mcal_moments
            gc.collect()
            save_obj(output_FLASK+'chunk_3_smoothed_seed_'+str(seed+1),['Done'])
    
    if compute_moments:
        print ('computing moments')
        mcal_moments = moments_map(conf)
        
        # smooth maps ******

        #mcal_moments.transform_and_smooth('convergence_noiseless_','g1','g2', shear = True, tomo_bins = [0,1,2,3], overwrite = False)
        mcal_moments.transform_and_smooth('convergence','e1','e2', shear = True, tomo_bins = [0,1,2,3], overwrite = False , skip_conversion_toalm = True)      
        mcal_moments.transform_and_smooth('noise','e1r','e2r', shear = True, tomo_bins = [0,1,2,3], overwrite = False , skip_conversion_toalm = True) 
        mcal_moments.transform_and_smooth('density','d', shear = False, tomo_bins = [0,1,2,3,4], overwrite = False , skip_conversion_toalm = True)    
        mcal_moments.transform_and_smooth('randoms','r', shear = False, tomo_bins = [0,1,2,3,4], overwrite = False , skip_conversion_toalm = True)                 
    
            
        del mcal_moments.fields 
        gc.collect()
    
        # We don't de-noise here - we do it at posteriori when assemplying the covariance.
        
        #mcal_moments.compute_moments( label_moments='noiseless_kEkE', field_label1 ='convergence_noiseless_kE', denoise1 = None,  tomo_bins1 = [0,1,2,3])
        #mcal_moments.compute_moments( label_moments='noiseless_kBkB', field_label1 ='convergence_noiseless_kB', denoise1 = None,  tomo_bins1 = [0,1,2,3])
    
        mcal_moments.compute_moments( label_moments='kEkE', field_label1 ='convergence_kE',  tomo_bins1 = [0,1,2,3])
        mcal_moments.compute_moments( label_moments='kBkB', field_label1 ='convergence_kB',  tomo_bins1 = [0,1,2,3])
    
        # k NN needs to be subtracted for corrections to the third moments ***
        mcal_moments.compute_moments( label_moments='kEkN', field_label1 ='convergence_kE', field_label2 = 'noise_kE', tomo_bins1 = [0,1,2,3])
        mcal_moments.compute_moments( label_moments='kNkN', field_label1 ='noise_kE', field_label2 = 'noise_kE', tomo_bins1 = [0,1,2,3])
        mcal_moments.compute_moments( label_moments='kNBkNB', field_label1 ='noise_kB', field_label2 = 'noise_kB', tomo_bins1 = [0,1,2,3])
        mcal_moments.compute_moments( label_moments='kNkE', field_label2 ='convergence_kE', field_label1 = 'noise_kE',  tomo_bins1 = [0,1,2,3])
    
        mcal_moments.compute_moments( label_moments='dKK', field_label1 ='density_kE', field_label2 = 'convergence_kE',  tomo_bins1 = [0,1,2,3,4], tomo_bins2 = [0,1,2,3])
        mcal_moments.compute_moments( label_moments='Kdd', field_label2 ='density_kE', field_label1 = 'convergence_kE',   tomo_bins2 = [0,1,2,3,4], tomo_bins1 = [0,1,2,3])
        mcal_moments.compute_moments( label_moments='dkNkN', field_label2 ='noise_kE', field_label1 = 'density_kE',  tomo_bins1 = [0,1,2,3,4], tomo_bins2 = [0,1,2,3])
        mcal_moments.compute_moments( label_moments='nKK', field_label2 ='convergence_kE', field_label1 = 'randoms_kE',  tomo_bins1 = [0,1,2,3,4], tomo_bins2 = [0,1,2,3])
        mcal_moments.compute_moments( label_moments='nkNkN', field_label2 ='noise_kE', field_label1 = 'randoms_kE',  tomo_bins1 = [0,1,2,3,4], tomo_bins2 = [0,1,2,3])

        mcal_moments.compute_moments( label_moments='kNdd', field_label1 ='noise_kE', field_label2 = 'density_kE',  tomo_bins1 = [0,1,2,3], tomo_bins2 = [0,1,2,3,4])


        mcal_moments.compute_moments( label_moments='knn', field_label1 ='convergence_kE', field_label2 = 'randoms_kE',  tomo_bins1 = [0,1,2,3], tomo_bins2 = [0,1,2,3,4])
        mcal_moments.compute_moments( label_moments='kNnn', field_label1 ='noise_kE', field_label2 = 'randoms_kE',  tomo_bins1 = [0,1,2,3], tomo_bins2 = [0,1,2,3,4])
        mcal_moments.compute_moments( label_moments='dd', field_label2 ='density_kE', field_label1 = 'density_kE',  tomo_bins1 = [0,1,2,3,4])
        mcal_moments.compute_moments( label_moments='nn', field_label2 ='randoms_kE', field_label1 = 'randoms_kE',  tomo_bins1 = [0,1,2,3,4])
        mcal_moments.compute_moments( label_moments='dnn', field_label1 ='density_kE', field_label2 = 'randoms_kE',  tomo_bins1 = [0,1,2,3,4])
        mcal_moments.compute_moments( label_moments='ndd', field_label1 ='randoms_kE', field_label2 = 'density_kE',  tomo_bins1 = [0,1,2,3,4])


    
        try:
            del mcal_moments.smoothed_maps
            gc.collect()
        except:
            pass
        
        save_obj(output_FLASK+'moments_seed_'+str(seed+1),mcal_moments)
        # delete the folder.
        os.system('rm -r '+output_FLASK+'/output_'+str(i+1))
if __name__ == '__main__':
    runstodo = []
    chunks_to_do =[]
    for i in range(0,n_FLASK_real):
        if not os.path.exists(output_FLASK+'chunk_1_smoothed_seed_'+str(i+1)+'.pkl'):
            runstodo.append(i)
            chunks_to_do.append(1)
        if not os.path.exists(output_FLASK+'chunk_2_smoothed_seed_'+str(i+1)+'.pkl'):
            runstodo.append(i)
            chunks_to_do.append(2)
        if not os.path.exists(output_FLASK+'chunk_3_smoothed_seed_'+str(i+1)+'.pkl'):
            runstodo.append(i)
            chunks_to_do.append(3)
        if not os.path.exists(output_FLASK+'moments_seed_'+str(i+1)+'.pkl'):
            if compute_moments:
                runstodo.append(i)
                chunks_to_do.append(0)
    #print  (runstodo)
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
