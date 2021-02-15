import pickle
import h5py as h5
import numpy as np
import healpy as hp
from Moments_analysis import convert_to_pix_coord


def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()



mute = h5.File('/global/cscratch1/sd/troxel/cats_des_y3/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT.h5','r')

rar = np.array(mute['randoms']['redmagic']['combined_sample_fid']['ra'])
decr = np.array(mute['randoms']['redmagic']['combined_sample_fid']['dec'])
mute.close()

mask_DESY3 = np.zeros(hp.nside2npix(nside))
pix1 = convert_to_pix_coord(rar,decr, nside=nside)
unique_pix1, idx1, idx_rep1 = np.unique(pix1, return_index=True, return_inverse=True)
mask_DESY3[unique_pix1] += np.bincount(idx_rep1, weights=np.ones(len(pix1)))
mask_DESY3 = mask_DESY3!=0.

save_obj('mask_DES_y3',mask_DESY3)