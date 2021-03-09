'''
This code takes FLASK output (convergence, shear, density fields) and add DES Y3 shape noise. It create maps of the shear field that needs to be used to compute the moments convariance.
'''
import gc
import pyfits as pf
import pickle
import numpy as np
from mpi4py import MPI 
import healpy as hp
import os
import copy
from Moments_analysis import convert_to_pix_coord, IndexToDeclRa, apply_random_rotation, addSourceEllipticity
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
This code takes the output of FLASK maps. It generates simulated des y3 like catalogs, adding shape noise and weights from the fiducial des y3 catalog on data. it also saves shear maps and desnity maps for the lenses. 


srun --nodes=20 --tasks-per-node=3 --cpus-per-task=20 --cpu-bind=cores --mem=110GB python convert_flask2maps.py
'''

  
mask_DES_y3 = load_obj('/global/homes/m/mgatti/Mass_Mapping/Moments_analysis/Covariance/mask_DES_y3')
nside = 1024
n_FLASK_real = 100
FLASK_path = '/global/cscratch1/sd/faoli/flask_desy3/4096/'
output = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/flask_tests/'

def make_maps(seed):
    density_full_maps = dict()
    density_maps = dict()
    lenses_maps = dict()
    random_maps = dict()
    sources_maps = dict()
    sources_cat = dict()
    # load the mcal catalog already divided into bins.
    
    FLASK_lens = pf.open(FLASK_path+'seed'+str(seed+1)+'/lens-catalog.fits.gz')

    lens_ra = FLASK_lens[1].data['ra']
    lens_dec = FLASK_lens[1].data['dec']
    lens_bin = FLASK_lens[1].data['galtype']
    # loop over tomographic bins (lenses)
    for i in range(5):
        mask_z = lens_bin == i+1
        lenses_maps[i] = np.zeros(hp.nside2npix(nside))
        pix1 = convert_to_pix_coord(lens_ra[mask_z],lens_dec[mask_z], nside=nside)
        unique_pix1, idx1, idx_rep1 = np.unique(pix1, return_index=True, return_inverse=True)
        lenses_maps[i][unique_pix1] += np.bincount(idx_rep1, weights=np.ones(len(pix1)))

        number_of_lenses = np.int(np.sum(lenses_maps[i]))
        lenses_maps[i][mask_DES_y3] = (lenses_maps[i][mask_DES_y3]-np.mean(lenses_maps[i][mask_DES_y3]))/np.mean(lenses_maps[i][mask_DES_y3])
        lenses_maps[i][~mask_DES_y3] =0.
    
        random_maps[i] = np.zeros(hp.nside2npix(nside))
        indexes = np.arange(len(lenses_maps[i]))[mask_DES_y3]
        pix_randoms = indexes[np.random.randint(0,len(indexes),number_of_lenses)]
        unique_pix1, idx1, idx_rep1 = np.unique(pix_randoms, return_index=True, return_inverse=True)
        random_maps[i][unique_pix1] += np.bincount(idx_rep1, weights=np.ones(len(pix_randoms)))
        random_maps[i][mask_DES_y3] = (random_maps[i][mask_DES_y3]-np.mean(random_maps[i][mask_DES_y3]))/np.mean(random_maps[i][mask_DES_y3])
        random_maps[i][~mask_DES_y3] =0.
        
        mute = pf.open(FLASK_path+'seed'+str(seed+1)+'/map-f{0}z{0}.fits'.format(i+1))
        density = mute[1].data['signal'].reshape(196608*1024)
        density_full_maps[i]  = hp.alm2map(hp.map2alm(density,lmax = 2048),nside = 1024)
        density_maps[i]  = copy.copy(density_full_maps[i])
        density_maps[i][~mask_DES_y3] = 0
    
        

    
    
    del pix1
    del lens_ra
    del lens_dec
    del lens_bin
    del FLASK_lens
    gc.collect()
    
    # loop over tomographic bins:
    mcal_catalog = load_obj('/project/projectdirs/des/mass_maps/Maps_final/data_catalogs_weighted')
    
    for i in range(4):
        # load_FLASK
        FLASK_shear = pf.open(FLASK_path+'seed'+str(seed+1)+'/kappa-gamma-f10z'+str(i+1)+'.fits')
        k = FLASK_shear[1].data['signal'].reshape(196608*1024)
        g1 = FLASK_shear[1].data['Q-pol'].reshape(196608*1024)
        g2 = FLASK_shear[1].data['U-pol'].reshape(196608*1024)
        
        g1_1024_full = hp.alm2map(hp.map2alm(g1,lmax = 2048),nside = 1024)
        g2_1024_full = hp.alm2map(hp.map2alm(g2,lmax = 2048),nside = 1024)
        
        g1_1024 = copy.copy(g1_1024_full)
        g2_1024 = copy.copy(g2_1024_full)
        g1_1024[~mask_DES_y3] = 0
        g2_1024[~mask_DES_y3] = 0
        
        del FLASK_shear

        dec1 = mcal_catalog[i]['dec']
        ra1 = mcal_catalog[i]['ra']
        w = mcal_catalog[i]['w']
        print (len(w))
        
        es1,es2 = apply_random_rotation(mcal_catalog[i]['e1'], mcal_catalog[i]['e2'])
        pix = convert_to_pix_coord(ra1,dec1, nside=4096)
        g1 = g1[pix]
        g2 = g2[pix]
        kk = k[pix]
        x1,x2 = addSourceEllipticity({'shear1':g1,'shear2':g2},{'e1':es1,'e2':es2},es_colnames=("e1","e2"))
        
        # repeat the rotation such that we have a different noise realisation for the noise map.
        es1,es2 = apply_random_rotation(mcal_catalog[i]['e1'], mcal_catalog[i]['e2'])
        
        k_orig = np.zeros(hp.nside2npix(nside))
        n_map = np.zeros(hp.nside2npix(nside))
        nn = np.zeros(hp.nside2npix(nside))
        e1_map = np.zeros(hp.nside2npix(nside))
        e2_map = np.zeros(hp.nside2npix(nside))
        g1_map = np.zeros(hp.nside2npix(nside))
        g2_map = np.zeros(hp.nside2npix(nside))
    
        e1_map_rndm = np.zeros(hp.nside2npix(nside))
        e2_map_rndm = np.zeros(hp.nside2npix(nside))

        pix = convert_to_pix_coord(ra1,dec1, nside=nside)
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        n_map[unique_pix] += np.bincount(idx_rep, weights=w)
        nn[unique_pix] += np.bincount(idx_rep, weights=np.ones(len(w)))
        
        k_orig[unique_pix] += np.bincount(idx_rep, weights= kk*w)
        e1_map[unique_pix] += np.bincount(idx_rep, weights= x1*w)
        e2_map[unique_pix] += np.bincount(idx_rep, weights= x2*w)
        g1_map[unique_pix] += np.bincount(idx_rep, weights= g1*w)
        g2_map[unique_pix] += np.bincount(idx_rep, weights= g2*w)
        e1_map_rndm[unique_pix] += np.bincount(idx_rep, weights= es1*w)
        e2_map_rndm[unique_pix] += np.bincount(idx_rep, weights= es2*w)
        
        mask_sims = n_map != 0.
       
        k_orig[mask_sims] = k_orig[mask_sims]/n_map[mask_sims]
        g1_map[mask_sims]  = g1_map[mask_sims]/(n_map[mask_sims])
        g2_map[mask_sims]  = g2_map[mask_sims]/(n_map[mask_sims])
        e1_map[mask_sims]  = e1_map[mask_sims]/(n_map[mask_sims])
        e2_map[mask_sims] =  e2_map[mask_sims]/(n_map[mask_sims])
    
        e1_map_rndm[mask_sims]  = e1_map_rndm[mask_sims]/(n_map[mask_sims])
        e2_map_rndm[mask_sims] =  e2_map_rndm[mask_sims]/(n_map[mask_sims])
        
        sources_cat[i] = {'ra': ra1, 'dec': dec1,'w':w, 'e1':x1,'e2':x2}
        sources_maps[i] = {'k_orig_map':k_orig,'g1_field_full':g1_1024_full,'g2_field_full':g2_1024_full,'g1_field':g1_1024,'g2_field':g2_1024,'g1_map':g1_map,'g2_map':g2_map,'e1_map':e1_map,'e2_map':e2_map,'e1r_map':e1_map_rndm,'e2r_map':e2_map_rndm,'N_source_map':nn,'N_source_mapw':n_map}
        
        del nn
        del n_map
        del e1_map_rndm
        del e2_map_rndm
        del k_orig
        del g1_map
        del g2_map
        del e1_map
        del e2_map
        gc.collect()
        
    save_obj(output+'seed_'+str(seed+1),{'density_full':density_full_maps,'density':density_maps,'randoms':random_maps,'lenses':lenses_maps,'sources':sources_maps})
    save_obj(output+'seed_cat_'+str(seed+1),sources_cat)
      
if __name__ == '__main__':
    runstodo=[]
    for i in range(0,n_FLASK_real):
        if not os.path.exists(output+'seed_'+str(i+1)+'.pkl'):
            runstodo.append(i)
    run_count=0
    while run_count<len(runstodo):
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        try:
            make_maps(runstodo[run_count+comm.rank])
        except:
            pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 


    
