'''
This class takes as input maps of the shear field and compute the smoothed moments (2nd and third order) out of them.
'''

import pyfits as pf
import numpy as np
import os
import copy
import gc
import healpy as hp
import scipy
import scipy.special  
from .healpy_utils import IndexToDeclRa,convert_to_pix_coord
from .smoothing_utils import almxfl
from astropy.table import Table
import multiprocessing
from functools import partial
try:
    import pys2let
    from pys2let import *
except:
    print ('missing pys2let')
try:
    import pywph as pw
except:
    print ('missing pywph')

class moments_map(object):
    def __init__(self,conf = {'output_folder': './'}):
        '''
        Initialise the moments_map object.
        '''
        self.conf = conf
        self.smoothed_maps = dict()
        self.fields = dict()
        self.moments = dict()
        try:
            if not os.path.exists(self.conf['output_folder']):
                os.mkdir((self.conf['output_folder']))
        except:
            pass
        try:
            if not os.path.exists((self.conf['output_folder'])+'/smoothed_maps/'):
                os.mkdir((self.conf['output_folder'])+'/smoothed_maps/')
        except:
            pass
    def add_map(self, map_, field_label ='', tomo_bin = 0):
        '''
        Add a map_ to the class 'fields' entry. need to specify
        the field label and the tomo_bin.
        '''
        if field_label in self.fields.keys():
            self.fields[field_label][tomo_bin] = copy.deepcopy(map_)
        else:
            self.fields[field_label] = dict()
            self.fields[field_label][tomo_bin] = copy.deepcopy(map_)
            
            
    def transform_and_smooth(self, output_label = '', field_label1 = '', field_label2 = None, shear = True, tomo_bins = [0], overwrite = False, skip_loading_smoothed_maps = False, skip_conversion_toalm = False):
        '''
        It takes 1 (2) field(s), compute the harmonic coefficients, multiply a top hat function, a gives back the smoothed map. If shear = True, it assumes the field1,field2 = e1,e2 and makes the conversion to k_E and k_B. The smoothed maps are also saved.
        n.b.: the two following keywordare normally set to False except when computing the covariance. it helps
        saving memory.
        skip_loading_smoothed_maps -> if the smoothed map already exists, it doesn't load it.
        skip_conversion_toalm  -> this can be set True if all the smoothed maps already exist. it skips the the conversion to alm from the field maps.
        '''
        
        lmax = self.conf['lmax']
        nside = self.conf['nside']
        ll = ['kE','kB']
        for ix in range(2):
            # it initialises the smoothed_maps dictionaries
            self.smoothed_maps[output_label+'_'+ll[ix]] = dict()
                
        # loop over bins.
        for binx in tomo_bins:
            for ix in range(2):
                self.smoothed_maps[output_label+'_'+ll[ix]][binx] = dict()
                
            if self.conf['verbose']:
                print(output_label,binx)
            
            alms_container = []
            
            ell, emm = hp.Alm.getlm(lmax=lmax)
            if not skip_conversion_toalm:
                # check if we're transforming a shear field or density field. 
                if field_label2 != None:
                    KQU_masked_maps = [self.fields[field_label1][binx],self.fields[field_label1][binx],self.fields[field_label2][binx]]
                else:
                    KQU_masked_maps = self.fields[field_label1][binx]
                    
                
                if (shear) and (len(KQU_masked_maps) == 3):
                    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!
                    
                    # E modes
                    alms_container.append(alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
                    # B modes
                    alms_container.append(alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
                    alms_container[0][ell == 0] = 0.0
                    alms_container[1][ell == 0] = 0.0
                    alms_container[0][ell == 1] = 0.0
                    alms_container[1][ell == 1] = 0.0
                    del alms
        
                else:
                    alms_container.append(hp.map2alm(KQU_masked_maps, lmax=lmax, pol=False))
                
                del KQU_masked_maps
                
                gc.collect()
            else:
                if field_label2 != None:
                    alms_container = [None, None]
                else:
                    alms_container = [None]
                
                
                
            
         
            for ix, almx in enumerate(alms_container):
                
                # loop over the smoothing scales
                for j, sm in enumerate(self.conf['smoothing_scales']):
                    if self.conf['verbose']:
                        print (sm)
                    path = self.conf['output_folder']+'/smoothed_maps/'+output_label+'_'+ll[ix]+'_bin_'+str(binx)+'_sm_'+str(sm)+'.fits'
                    # check if the map has already been computed
                    if (not os.path.exists(path)) or (overwrite):
                        
                        #conversion to radiant
                        sm_rad =(sm/60.)*np.pi/180.  
    
                        # smoothing kernel (top-hat)
                        A = 1./(2*np.pi*(1.-np.cos(sm_rad)))
                        B = (2*np.pi/(2.*ell+1.0))
                        fact = -B*(scipy.special.eval_legendre(ell+1,np.cos(sm_rad))-scipy.special.eval_legendre(ell-1,np.cos(sm_rad)))*A;
                        fact[0] = 1
    
                        # multiply the field alms with the smoothing kernel
                        alms1 = almxfl(almx,fact,inplace=False)
                        # convert bak to map
                        mapaa = hp.alm2map(alms1, nside= self.conf['nside'], lmax=self.conf['lmax'], pol=False)
                        del alms1
                
                        # save the map 
                        if os.path.exists(path):
                            os.remove(path)
                        fits_f = Table()
                        fits_f['map'] = np.array(mapaa) #CHANGED!!#[mute_mask])
                        fits_f.write(path)
                        self.smoothed_maps[output_label+'_'+ll[ix]][binx][sm] = path
                    else:
                        if not skip_loading_smoothed_maps:
                            # load the map
                            mute = pf.open(path,memmap=False)
                            mapaa = copy.copy(mute[1].data['map'])
                        else:
                            mapaa = None
                        self.smoothed_maps[output_label+'_'+ll[ix]][binx][sm] = path
                    if not skip_loading_smoothed_maps:
                        self.smoothed_maps[output_label+'_'+ll[ix]][binx][sm] = copy.deepcopy(mapaa)
                        del mapaa
            del alms_container
            gc.collect()

    def transform_and_smooth_sp(self, output_label = '', field_label1 = '', field_label2 = None, shear = True, tomo_bins = [0], overwrite = False, skip_loading_smoothed_maps = False, skip_conversion_toalm = False):
        '''
        It takes 1 (2) field(s), compute the harmonic coefficients, multiply a top hat function, a gives back the smoothed map. If shear = True, it assumes the field1,field2 = e1,e2 and makes the conversion to k_E and k_B. The smoothed maps are also saved.
        n.b.: the two following keywordare normally set to False except when computing the covariance. it helps
        saving memory.
        skip_loading_smoothed_maps -> if the smoothed map already exists, it doesn't load it.
        skip_conversion_toalm  -> this can be set True if all the smoothed maps already exist. it skips the the conversion to alm from the field maps.
        '''
        
        lmax = self.conf['lmax']
        nside = self.conf['nside']
        ll = ['kE','kB']
        for ix in range(2):
            # it initialises the smoothed_maps dictionaries
            self.smoothed_maps[output_label+'_'+ll[ix]] = dict()
                
        # loop over bins.
        for binx in tomo_bins:
            for ix in range(2):
                self.smoothed_maps[output_label+'_'+ll[ix]][binx] = dict()
                
            if self.conf['verbose']:
                print(output_label,binx)
            
            alms_container = []
            
            ell, emm = hp.Alm.getlm(lmax=lmax-1)
            if not skip_conversion_toalm:
                # check if we're transforming a shear field or density field. 
                if field_label2 != None:
                    KQU_masked_maps = [self.fields[field_label1][binx],self.fields[field_label1][binx],self.fields[field_label2][binx]]
                else:
                    KQU_masked_maps = self.fields[field_label1][binx]
                    
                
                if (shear) and (len(KQU_masked_maps) == 3):
                    alms = hp.map2alm(KQU_masked_maps, lmax=lmax-1, pol=True)  # Spin transform!
                    
                    # E modes
                    alms_container.append(alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
                    # B modes
                    alms_container.append(alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
                    alms_container[0][ell == 0] = 0.0
                    alms_container[1][ell == 0] = 0.0
                    alms_container[0][ell == 1] = 0.0
                    alms_container[1][ell == 1] = 0.0
                    del alms
        
                else:
                    alms_ = hp.map2alm(KQU_masked_maps, lmax=lmax-1, pol=False)
                    alms_container.append(alms_)
                    alms_container.append(alms_*0.)
                
                del KQU_masked_maps
                
                gc.collect()
            else:
                if field_label2 != None:
                    alms_container = [None, None]
                else:
                    alms_container = [None]
                
                
                
            
         
            for ix, almx in enumerate(alms_container):
                    J_min = self.conf['J_min']
                    B = self.conf['B']
                    J = pys2let_j_max(2, lmax, J_min)


                    # Read healpix map and compute alms. Thi ssuppresses all the power above L.
                    # it is usually not needed - when computing hte convergence field from the shear field, the map is already band limited.
                    f_wav_lm, f_scal_lm = analysis_axisym_lm_wav(almx, 2, lmax, J_min)
                    
                    for sm in range(J - J_min + 1):
                        flm = f_wav_lm[:, sm].ravel()              
                        path = self.conf['output_folder']+'/smoothed_maps/'+output_label+'_'+ll[ix]+'_bin_'+str(binx)+'_sm_'+str(sm)+'.fits'
                        if (not os.path.exists(path)) or (overwrite):
                            mapaa = hp.alm2map(flm, nside=nside, lmax=lmax - 1)
                            # save the map 
                            if os.path.exists(path):
                                os.remove(path)
                            fits_f = Table()
                            fits_f['map'] = np.array(mapaa) #CHANGED!!#[mute_mask])
                            fits_f.write(path)
                        else:
                            if not skip_loading_smoothed_maps:
                                # load the map
                                mute = pf.open(path,memmap=False)
                                mapaa = copy.copy(mute[1].data['map'])
                            else:
                                mapaa = None
                        if not skip_loading_smoothed_maps:
                            self.smoothed_maps[output_label+'_'+ll[ix]][binx][sm] = path #copy.deepcopy(mapaa)
                            del mapaa
            del alms_container
            gc.collect()

         
        
        
        
        
    def transform_and_smooth_sp_dir(self, output_label = '', field_label1 = '', field_label2 = None, shear = True, tomo_bins = [0], overwrite = False, skip_loading_smoothed_maps = False, skip_conversion_toalm = False):
        '''
        It takes 1 (2) field(s), compute the harmonic coefficients, multiply a top hat function, a gives back the smoothed map. If shear = True, it assumes the field1,field2 = e1,e2 and makes the conversion to k_E and k_B. The smoothed maps are also saved.
        n.b.: the two following keywordare normally set to False except when computing the covariance. it helps
        saving memory.
        skip_loading_smoothed_maps -> if the smoothed map already exists, it doesn't load it.
        skip_conversion_toalm  -> this can be set True if all the smoothed maps already exist. it skips the the conversion to alm from the field maps.
        '''
        
        lmax = self.conf['lmax']
        nside = self.conf['nside']
        


        if field_label2 != None:
            ll = ['kE','kB']
            for ix in range(2):
                # it initialises the smoothed_maps dictionaries
                self.smoothed_maps[output_label+'_'+ll[ix]] = dict()
        else:
            self.smoothed_maps[output_label] = dict()
        # loop over bins.
        for binx in tomo_bins:
            if field_label2 != None:
                for ix in range(2):
                    self.smoothed_maps[output_label+'_'+ll[ix]][binx] = dict()
            else:
                self.smoothed_maps[output_label][binx] = dict()
                
            if self.conf['verbose']:
                print(output_label,binx)
            
            alms_container = []
            
            ell, emm = hp.Alm.getlm(lmax=lmax-1)
            if not skip_conversion_toalm:
                # check if we're transforming a shear field or density field. 
                if field_label2 != None:
                    KQU_masked_maps = [self.fields[field_label1][binx],self.fields[field_label1][binx],self.fields[field_label2][binx]]
                else:
                    KQU_masked_maps = self.fields[field_label1][binx]
                    
                
                if (shear) and (len(KQU_masked_maps) == 3):
                    alms = hp.map2alm(KQU_masked_maps, lmax=lmax-1, pol=True)  # Spin transform!
                    
                    # E modes
                    alms_container.append(alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
                    # B modes
                    alms_container.append(alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
                    alms_container[0][ell == 0] = 0.0
                    alms_container[1][ell == 0] = 0.0
                    alms_container[0][ell == 1] = 0.0
                    alms_container[1][ell == 1] = 0.0
                    del alms
        
                else:
                    alms_ = hp.map2alm(KQU_masked_maps, lmax=lmax-1, pol=False)
                    alms_container.append(alms_)
                    #alms_container.append(alms_*0.)
                
                del KQU_masked_maps
                
                gc.collect()
            else:
                if field_label2 != None:
                    alms_container = [None, None]
                else:
                    alms_container = [None]
                
                
                
            
         
            for ix, almx in enumerate(alms_container):
                    J_min = self.conf['J_min']
                    B = self.conf['B']
                    N = self.conf['N']
                    upsample = 1
                    spin = 0
                    
                    L = self.conf['nside']*2


                    J = pys2let_j_max(2, lmax, J_min)


                    # Read healpix map and compute alms. Thi ssuppresses all the power above L.
                    # it is usually not needed - when computing hte convergence field from the shear field, the map is already band limited.
                    f_wav, f_scal = analysis_lm2wav(almx, B, L, J_min, N, spin, upsample)
                    

                    for j in range(J_min, J):
                        for n in range(0, N):
                            # Retreive the boundaries and positions of the right wavelet scale in the giant f_wav array!
                            offset, bandlimit, nelem, nelem_wav = wav_ind(j, n, B, L, N, J_min, upsample)

                            L = bandlimit
                            thetas, phis = mw_sampling(L)
                            ntheta = len(thetas)
                            nphi = len(phis)
                            # Convert the input MW 1D array into 2D array with rows for theta and columns for phi. As simple as that!
                            arr = f_wav[offset : offset + nelem].reshape((ntheta, nphi))
                            if field_label2 != None:
                                self.smoothed_maps[output_label+'_'+ll[ix]][binx]['{0}_{1}'.format(j,n)] = arr
                
                            else:
                                self.smoothed_maps[output_label][binx]['{0}_{1}'.format(j,n)] = arr

                            

            del alms_container
            gc.collect()
            
            
            
            
            
    def compute_moments(self, label_moments='', field_label1 ='', field_label2  =None, denoise1 = None, denoise2 = None, tomo_bins1 = [0,1,2,3], tomo_bins2 = None):
        '''
        computes second and third moments given different fields.
        if field_label2 is provided, it will compute the moments cross moments between fields.
        The denoise option is to subtract shape noise/shot noise.
        
        second moments = field_1 field_2
        third moments field_1 field_2 field_3
        '''
        moments_mute = dict()


        # the follwoing bit is for the cases
        # where we have ddk correlations.
        
        if tomo_bins2 == None:
            tomo_bins2 = copy.deepcopy(tomo_bins1)
        if (field_label2 != field_label1) and (field_label2 != None):
            do_all_matrix = True
        else:
            do_all_matrix = False

            denoise2 = copy.copy(denoise1)
            field_label2 = copy.copy(field_label1)
            
        # we start with the second moments computation
        for i,bin1 in enumerate(tomo_bins1):
            for j,bin2 in enumerate(tomo_bins2):
                moments_2 = np.zeros((len(self.conf['smoothing_scales'])))
                moments_3 = np.zeros((len(self.conf['smoothing_scales'])))
                
                # this allows you to skip the computation of some moments due to simmetry reasons
                if (not do_all_matrix) and (j>=i):
                    compute_it = True
                elif (not do_all_matrix) and (j<i):
                    compute_it = False
                elif (do_all_matrix): 
                    compute_it = True
                if compute_it:
                    for kk, sm in enumerate(self.conf['smoothing_scales']):
                        
                        mute = pf.open(self.smoothed_maps[field_label1][bin1][sm],memmap=False)
                        map_signal1 = copy.copy(mute[1].data['map'])
                        mute = pf.open(self.smoothed_maps[field_label2][bin2][sm],memmap=False)
                        map_signal2 = copy.copy(mute[1].data['map'])
                        '''
                        map_signal1 = self.smoothed_maps[field_label1][bin1][sm]
                        map_signal2 = self.smoothed_maps[field_label2][bin2][sm]
                        '''
                        
                        if (denoise1  != None) and (i==j) and (denoise1 == denoise2):
                            mute = pf.open(self.smoothed_maps[denoise1][bin1][sm],memmap=False)
                            map_rndm1 = copy.copy(mute[1].data['map'])
                            mute = pf.open(self.smoothed_maps[denoise2][bin2][sm],memmap=False)
                            map_rndm2 = copy.copy(mute[1].data['map'])
                            '''
                            map_rndm1 = self.smoothed_maps[denoise1][bin1][sm]
                            map_rndm2 = self.smoothed_maps[denoise2][bin2][sm]       
                            '''
                            moments_2[kk] = np.mean(map_signal1*map_signal2) - np.mean(map_rndm1*map_rndm2)
                        else:
                            moments_2[kk] = np.mean(map_signal1*map_signal2)
                            
                            

                    
                    moments_mute.update({'{0}_{1}'.format(bin1,bin2):moments_2})
            
            
        # symmetrise
        if not do_all_matrix:
            for i,bin1 in enumerate(tomo_bins1):
                for j,bin2 in enumerate(tomo_bins2):  
                    if i!=j:
                        if j<i:  
                            moments_mute['{0}_{1}'.format(bin1,bin2)]= copy.deepcopy(moments_mute['{1}_{0}'.format(bin1,bin2)])


            
        # we proceed with the third moments computation  
        for i,bin1 in enumerate(tomo_bins1):
            for j,bin2 in enumerate(tomo_bins2): 
                for zxk,bin3 in enumerate(tomo_bins2):
                    
                    # this allows you to skip the computation of some moments due to simmetry reasons
                    if (not do_all_matrix) and (zxk >=j) and (j>=i):
                        compute_it = True
                    elif (do_all_matrix) and (zxk >=j):
                        compute_it = True
                    else:
                        compute_it = False
                    if compute_it:
                        moments_3 = np.zeros(len(self.conf['smoothing_scales']))
                        for kk,sm in enumerate(self.conf['smoothing_scales']):
                            
                            mute = pf.open(self.smoothed_maps[field_label1][bin1][sm],memmap=False)
                            map_tbsm1 = copy.copy(mute[1].data['map'])
                            mute = pf.open(self.smoothed_maps[field_label2][bin2][sm],memmap=False)
                            map_tbsm2 = copy.copy(mute[1].data['map'])
                            mute = pf.open(self.smoothed_maps[field_label2][bin3][sm],memmap=False)
                            map_tbsm3 = copy.copy(mute[1].data['map'])
                        
                            '''
                            map_tbsm1 = self.smoothed_maps[field_label1][bin1][sm]
                            map_tbsm2 = self.smoothed_maps[field_label2][bin2][sm]
                            map_tbsm3 = self.smoothed_maps[field_label2][bin3][sm]
                            '''
                            moments_3[kk] = np.mean(map_tbsm1*map_tbsm2*map_tbsm3)
                        
                        moments_mute.update({'{0}_{1}_{2}'.format(bin1,bin2,bin3):moments_3})

        # symmetrise
        for i,bin1 in enumerate(tomo_bins1):
            for j,bin2 in enumerate(tomo_bins2): 
                for zxk,bin3 in enumerate(tomo_bins2):

                    binss = [bin1,bin2,bin3]
                    ordi =np.argsort([i,j,zxk])

                    
                    if (zxk <j) and (j>=i): 
                        moments_mute['{0}_{1}_{2}'.format(bin1,bin2,bin3)] = copy.deepcopy(moments_mute['{0}_{1}_{2}'.format(binss[ordi[0]],binss[ordi[1]],binss[ordi[2]])])
                    if not do_all_matrix:
                        if (zxk >=j) and (j<i):
                            moments_mute['{0}_{1}_{2}'.format(bin1,bin2,bin3)] = copy.deepcopy(moments_mute['{0}_{1}_{2}'.format(binss[ordi[0]],binss[ordi[1]],binss[ordi[2]])])
                        
                        if (zxk <j) and (j<i):
                            moments_mute['{0}_{1}_{2}'.format(bin1,bin2,bin3)] = copy.deepcopy(moments_mute['{0}_{1}_{2}'.format(binss[ordi[0]],binss[ordi[1]],binss[ordi[2]])])
        self.moments.update({label_moments: moments_mute})
        
        
        
        
        
    # M&M moments
    def compute_moments_gen(self, label_moments='', field_label1='', field_label2=None, denoise1=None, denoise2=None, tomo_bins1=[0, 1, 2, 3], tomo_bins2=None):
        '''
        computes second and third moments given different fields.
        if field_label2 is provided, it will compute the moments cross moments between fields.
        The denoise option is to subtract shape noise/shot noise.
        second moments = field_1 field_2
        third moments field_1 field_2 field_3
        self.moments will be a dictionary of
        (redshift bins for 2nd moments) : rank 2 tensor of all possible sm combinations
        (redshift bins for 3rd moments) : rank 3 tensor for all possible sm combinations
        '''
        
        print (tomo_bins1,tomo_bins2)
        moments_mute = dict()

        '''
        "matrix" in "do_all_matrix" is the tensor of smoothing scales
        if theres more than 1 redshift bins, then the combinatorics are non trivial so you should always
        compute all redshift bin combos. I think you can probably skip some sums, but the sums for the
        means are not the bottleneck, so it's ok
        '''
        if tomo_bins2 == None:
            tomo_bins2 = copy.deepcopy(tomo_bins1)
        if (field_label2 != field_label1) and (field_label2 != None):
            do_all_matrix = True
        elif (len(tomo_bins1) > 1):
            do_all_matrix = True
        else:
            do_all_matrix = False

            denoise2 = copy.copy(denoise1)
            field_label2 = copy.copy(field_label1)

        # we start with the second moments computation
        for i, bin1 in enumerate(tomo_bins1):
            for j, bin2 in enumerate(tomo_bins2):
                # at every tomo1-tomo2, there is a moment for every sm1-sm2
                moments_2 = np.zeros(
                    (len(self.conf['smoothing_scales']), len(self.conf['smoothing_scales'])))

                for kk1, sm1 in enumerate(self.conf['smoothing_scales']):
                    for kk2, sm2 in enumerate(self.conf['smoothing_scales']):
                        if do_all_matrix or kk2 >= kk1:
                            mute = pf.open(self.smoothed_maps[field_label1][bin1][sm1],memmap=False)
                            map_signal1 = copy.copy(mute[1].data['map'])
                            mute = pf.open(self.smoothed_maps[field_label2][bin2][sm2],memmap=False)
                            map_signal2 = copy.copy(mute[1].data['map'])
                        
                        
                      
                            if (denoise1 != None) and (i == j) and (denoise1 == denoise2):
                                mute = pf.open(self.smoothed_maps[denoise1][bin1][sm1],memmap=False)
                                map_rndm1 = copy.copy(mute[1].data['map'])
                                mute = pf.open(self.smoothed_maps[denoise2][bin2][sm2],memmap=False)
                                map_rndm2 = copy.copy(mute[1].data['map'])

                                moments_2[kk1, kk2] = np.mean(
                                    map_signal1*map_signal2) - np.mean(map_rndm1*map_rndm2)
                            else:
                                #print (kk1,kk2,map_signal1,map_signal2)
                                moments_2[kk1, kk2] = np.mean(
                                    map_signal1*map_signal2)
                        else:
                            moments_2[kk1, kk2] = moments_2[kk2, kk1]

                    moments_mute.update(
                        {'{0}_{1}'.format(bin1, bin2): moments_2})

        # we proceed with the third moments computation
        for i, bin1 in enumerate(tomo_bins1):
            for j, bin2 in enumerate(tomo_bins2):
                for zxk, bin3 in enumerate(tomo_bins2):
                   
                    # at every tomo1-tomo2-tomo3, there is a moment for every sm1-sm2-sm3
                    moments_3 = np.zeros((
                        len(self.conf['smoothing_scales']), len(self.conf['smoothing_scales']), len(self.conf['smoothing_scales'])))
                    for kk1, sm1 in enumerate(self.conf['smoothing_scales']):
                        mute = pf.open(self.smoothed_maps[field_label1][bin1][sm1],memmap=False)
                        map_tbsm1 = copy.copy(mute[1].data['map'])
                        for kk2, sm2 in enumerate(self.conf['smoothing_scales']):
                            mute = pf.open(self.smoothed_maps[field_label2][bin2][sm2],memmap=False)
                            map_tbsm2 = copy.copy(mute[1].data['map'])
                                    
                            for kk3, sm3 in enumerate(self.conf['smoothing_scales']):
                                if do_all_matrix or (kk3 >= kk2 and kk2 >= kk1):
                                    
                                    mute = pf.open(self.smoothed_maps[field_label2][bin3][sm3],memmap=False)
                                    map_tbsm3 = copy.copy(mute[1].data['map'])                                                 
                                    
                                    
                                  
                                    moments_3[kk1, kk2, kk3] = np.mean(
                                            map_tbsm1*map_tbsm2*map_tbsm3)
                                else:
                                    foo = np.sort([kk1, kk2, kk3])
                                    moments_3[kk1, kk2,kk3] = moments_3[foo[0], foo[1], foo[2]]

                    moments_mute.update(
                        {'{0}_{1}_{2}'.format(bin1, bin2, bin3): moments_3})

        self.moments.update({label_moments: moments_mute})


        
        
        
        
        
        

    def compute_WHMP_S01(self,label_moments,field_label1,field_label2=None,denoise1= None,denoise2 = None, tomo_bins1 = [0,1], tomo_bins2 = None,min_d = 0.1):
        '''
        computes S01
        '''
        moments_mute = dict()


        # the follwoing bit is for the cases
        # where we have ddk correlations.

        if tomo_bins2 == None:
            tomo_bins2 = copy.deepcopy(tomo_bins1)
        if (field_label2 != field_label1) and (field_label2 != None):
            do_all_matrix = True
        else:
            do_all_matrix = False

            denoise2 = copy.copy(denoise1)
            field_label2 = copy.copy(field_label1)

        # we start with the second moments computation
        for i,bin1 in enumerate(tomo_bins1):
            for j,bin2 in enumerate(tomo_bins2):
                count =0
                for kk, sm1 in enumerate(self.conf['smoothing_scales']):
                        for kk1, sm2 in enumerate(self.conf['smoothing_scales']):
                            if kk1<=kk:
                                if (sm1-sm2)**2<min_d:
                                    count += 1
                moments_2 = np.zeros(count)

                # this allows you to skip the computation of some moments due to simmetry reasons
                u = 0
                scales = []
                for kk, sm1 in enumerate(self.conf['smoothing_scales']):
                        for kk1, sm2 in enumerate(self.conf['smoothing_scales']):
                            if kk1<=kk:
                                if (sm1-sm2)**2<min_d:
                                    
                                    mute = pf.open(self.smoothed_maps[field_label1][bin1][sm1],memmap=False)
                                    map_signal1 = copy.copy(mute[1].data['map'])
                                    mute = pf.open(self.smoothed_maps[field_label2][bin2][sm2],memmap=False)
                                    map_signal2 =  np.abs(copy.copy(mute[1].data['map']))
                                    
                                    
                                    
                                    '''
                                    map_signal1 = self.smoothed_maps[field_label1][bin1][sm1]
                                    map_signal2 = np.abs(self.smoothed_maps[field_label2][bin2][sm2])
                                    '''
                                    
                                    
                                    if (denoise1  != None) and (i==j) and (denoise1 == denoise2):
                                        '''
                                        map_rndm1 = self.smoothed_maps[denoise1][bin1][sm1]
                                        map_rndm2 = np.abs(self.smoothed_maps[denoise2][bin2][sm2])    
                                        '''
                                        mute = pf.open(self.smoothed_maps[denoise1][bin1][sm1],memmap=False)
                                        map_rndm1 = copy.copy(mute[1].data['map'])
                                        mute = pf.open(self.smoothed_maps[denoise2][bin2][sm2],memmap=False)
                                        map_rndm2 =  np.abs(copy.copy(mute[1].data['map']))
                                    
                                        moments_2[u] = np.mean(map_signal1*map_signal2) - np.mean(map_rndm1*map_rndm2)
                                    else:
                                        moments_2[u] = np.mean(map_signal1*map_signal2)
                                        
                                    scales.append(sm1+sm2)
                                    u += 1


                moments_mute.update({'{0}_{1}'.format(bin1,bin2):moments_2})


        self.moments.update({'S01_'+str(label_moments): moments_mute})



    def compute_moments_SS2(self, label_moments='', field_label1 ='', field_label2  =None, denoise1 = None, denoise2 = None, tomo_bins1 = [0,1,2,3], tomo_bins2 = None, scale_ref = 0):
        '''
        computes second and third moments given different fields.
        if field_label2 is provided, it will compute the moments cross moments between fields.
        The denoise option is to subtract shape noise/shot noise.

        second moments = field_1 field_2
        third moments field_1 field_2 field_3
        '''
        moments_mute = dict()


        # the follwoing bit is for the cases
        # where we have ddk correlations.

        if tomo_bins2 == None:
            tomo_bins2 = copy.deepcopy(tomo_bins1)
        if (field_label2 != field_label1) and (field_label2 != None):
            do_all_matrix = True
        else:
            do_all_matrix = False

            denoise2 = copy.copy(denoise1)
            field_label2 = copy.copy(field_label1)

        ss = np.array([scale_ref,scale_ref+1])
        ss = ss[(ss>=0) & (ss< len(self.conf['smoothing_scales']))]
        # we start with the second moments computation
        for i,bin1 in enumerate(tomo_bins1):
            for j,bin2 in enumerate(tomo_bins2):
               # bin2 = bin1
                moments_2 = np.zeros((len(ss)))
                moments_3 = np.zeros((len(ss)))

                # this allows you to skip the computation of some moments due to simmetry reasons
                if (not do_all_matrix) and (j>=i):
                    compute_it = True
                elif (not do_all_matrix) and (j<i):
                    compute_it = False
                elif (do_all_matrix): 
                    compute_it = True
                if compute_it:
                    for kk, sm in enumerate(ss):

                        #map_signal1 = np.sqrt(np.abs(self.smoothed_maps[field_label1][bin1][sm]))
                        #map_signal2 = np.sqrt(np.abs(self.smoothed_maps[field_label2][bin2][sm]))
                        mute = pf.open(self.smoothed_maps[field_label1][bin1][sm],memmap=False)
                        map_signal1 = np.sqrt(np.abs(copy.copy(mute[1].data['map'])))
                        mute = pf.open(self.smoothed_maps[field_label2][bin2][sm],memmap=False)
                        map_signal2 =   np.sqrt(np.abs(copy.copy(mute[1].data['map'])))
                        
                        if (denoise1  != None) and (i==j) and (denoise1 == denoise2):
                           # map_rndm1 = np.sqrt(np.abs(self.smoothed_maps[denoise1][bin1][sm]))
                           # map_rndm2 = np.sqrt(np.abs(self.smoothed_maps[denoise2][bin2][sm]))    
                            mute = pf.open(self.smoothed_maps[denoise1][bin1][sm],memmap=False)
                            map_rndm1 = np.sqrt(np.abs(copy.copy(mute[1].data['map'])))
                            mute = pf.open(self.smoothed_maps[denoise2][bin2][sm],memmap=False)
                            map_rndm2 =   np.sqrt(np.abs(copy.copy(mute[1].data['map'])))
                        
                            moments_2[kk] = np.mean(map_signal1*map_signal2) - np.mean(map_rndm1*map_rndm2)
                        else:
                            moments_2[kk] = np.mean(map_signal1*map_signal2)


                    moments_mute.update({'{0}_{1}'.format(bin1,bin2):moments_2})



        self.moments.update({'SS2_'+label_moments: moments_mute})


    def cut_patches(self, nside=512, nside_small=16, threshold = 0.0001):
        '''
        It takes every entry in the field catalog, and cuts them into squared patches using the healpy gnomview projection.
        delta is the side length of the square patch in degrees.
        patches are saved into the 'field_patches' dictionary.

        '''
        
        # initial guess for the size of the patch
        delta = np.sqrt(hp.nside2pixarea(nside_small, degrees = True))*4
        
        # test maps -----------------------------
        mask_small = hp.ud_grade(self.mask,nside_out=nside_small)
        map_test_uniform = np.ones(len(self.mask))
        mask_pix = mask_small!=0.


        # this is to know which part to mask ---
        map_indexes = np.arange(hp.nside2npix(nside_small))
        map_indexes_large = np.arange(hp.nside2npix(nside))
        dec_large,ra_large = IndexToDeclRa(map_indexes_large, nside,nest= False)
        pix_large_convert = convert_to_pix_coord(ra_large,dec_large, nside=nside_small)

        # these are the centers
        dec_,ra_ = IndexToDeclRa(map_indexes, nside_small,nest= False)
        pairs_ = np.vstack([ra_[mask_pix],dec_[mask_pix],map_indexes[mask_pix]])

        pairs = []

        count_area = 0

        #print ('delta: ',delta)
        print ('number of patches: ',len(map_indexes))
        pixels = np.int(delta/(hp.nside2resol(nside, arcmin=True)/60))
        res = hp.nside2resol(nside, arcmin=True)
        xsize=2**np.int(np.log2(pixels))

        print ('resolution [arcmin]: ',res)
        print ('pixels per side: ',xsize)
        print ('frame size [deg]: ',xsize*res/60.)
        
        
        
        
        # checks wich patch to keep
        '''
        agents = 20
        xlist = np.arange(pairs_.shape[1])
        pool = multiprocessing.Pool(processes=agents)
        pairs = pool.map(partial(project_,  res=res,pairs_ = pairs_,xsize = xsize,mask = self.mask, pix_large_convert=pix_large_convert,map_test_uniform=copy.deepcopy(map_test_uniform)), xlist)
        pool.close()
        pool.join()
  
        # remove None
        pairs1 = []
        for p in pairs:
            if p is not None:
                pairs1.append(p)

                
        self.fields_patches = dict()
 
        
        for key in self.fields.keys():
            self.fields_patches[key] = dict()
            for key2 in self.fields[key].keys():

                patches = []

                xlist = np.arange(len(pairs1))
                pool = multiprocessing.Pool(processes=agents)
                patches_ = pool.map(partial(project_2,  pairs = pairs1,xsize = xsize, field=copy.deepcopy(self.fields[key][key2]),res=res,pix_large_convert =pix_large_convert,nside_small=nside_small), xlist)

                pool.close()
                pool.join()
                self.fields_patches[key][key2] = patches_ 
    
    
    
        '''
        for i in frogress.bar(range(pairs_.shape[1])):
            ra_,dec_,i_ = pairs_[:,i]


            map_test_ = copy.deepcopy(self.mask)
            map_test_uniform_ = copy.deepcopy(map_test_uniform)
            mask_ = np.in1d(pix_large_convert,i_)
            map_test_[~mask_] = 0.
            map_test_uniform_[~mask_] = 0.
            m_ref = hp.gnomview(map_test_, rot=(ra_,dec_), xsize=xsize ,no_plot=True,reso=res,return_projected_map=True)

            if (1.*np.sum((m_ref.data).flatten())/np.sum(map_test_uniform_.flatten()))>threshold: # more than 20% of the stamp!
                pairs.append([ra_,dec_,i_])
                count_area += (res/60.)**2 *np.sum(m_ref)

        print ('TOTAL AREA: ' ,count_area)
        #print ('SHAPE: ',(m_ref.data).shape)
  


        self.fields_patches = dict()
 
        
        for key in self.fields.keys():
            self.fields_patches[key] = dict()
            for key2 in self.fields[key].keys():

                patches = []

                for i in frogress.bar(range(len(pairs))):
                    ra_,dec_,i_ = pairs[i]
                    
                    fieldc = copy.deepcopy(self.fields[key][key2])
                    mask_ = np.in1d(pix_large_convert,i_)
                    fieldc[~mask_] = 0.
                    mt1 = hp.gnomview(fieldc, rot=(ra_,dec_), xsize=xsize ,no_plot=True,reso=res,return_projected_map=True)

                    patches.append(mt1.data)
                    
 
                np.save(self.conf['output_folder']+'{0}_{1}'.format(key,key2),patches)
                self.patch_size = patches[0].shape
                self.fields_patches[key][key2] = self.conf['output_folder']+'{0}_{1}'.format(key,key2)
                

        
    def compute_moments_pywhm(self,label,field1,field2,denoise1=None,denoise2=None):
        M = self.patch_size[0]
        N = self.patch_size[1]
        tomo_bins = list(self.fields_patches[field1].keys())

        J = self.conf['J']
        j_min = self.conf['j_min']
        L =  self.conf['L']
        dn = 0

        self.moments_pywph[label] = dict()
        try:
            self.moments_pywph__[label] = dict()
        except:
            pass
        self.moments_pywph_indexes[label] = dict()
        #if 1==1:
        #        i = 0
        #        j = 3
        for i in tomo_bins:
            for j in tomo_bins:
                    
                #for k in range(len(self.fields_patches[field1][list(self.fields_patches[field1].keys())[0]])):

                    patch1 = np.load(self.fields_patches[field1][i]+'.npy',allow_pickle=True)
                    patch2 = np.load(self.fields_patches[field2][j]+'.npy',allow_pickle=True)
                    if (denoise1  != None):
                        npatch1 = np.load(self.fields_patches[denoise1][i]+'.npy',allow_pickle=True)
                        npatch2 = np.load(self.fields_patches[denoise2][j]+'.npy',allow_pickle=True)   

                    wph_op = pw.WPHOp(M, N, J, L=L,j_min=j_min, dn=dn, device='cpu')
                    wph = wph_op([patch1, patch2], cross=True, ret_wph_obj=True)
                    wph.to_isopar()
                    s00, s00_indices = wph.get_coeffs("S00")
                    s11, s11_indices = wph.get_coeffs("S11")
                    s01, s01_indices = wph.get_coeffs("S01")
                    c01, c01_indices = wph.get_coeffs("C01")
                    cphase, cphase_indices = wph.get_coeffs("Cphase")
                    c00, c00_indices = wph.get_coeffs("C00")
                    if (denoise1  != None):
                        wph = wph_op([npatch1, npatch2], cross=True, ret_wph_obj=True)
                        wph.to_isopar()
                        ns00, s00_indices = wph.get_coeffs("S00")
                        ns11, s11_indices = wph.get_coeffs("S11")
                        ns01, s01_indices = wph.get_coeffs("S01")
                        nc01, c01_indices = wph.get_coeffs("C01")
                        ncphase, cphase_indices = wph.get_coeffs("Cphase")
                        nc00, c00_indices = wph.get_coeffs("C00")

                            
                    '''
                    nm2 = []
                    nm3 = []
                    nm4 = []
                    nm_ = []
                    for j in range(j_min,J):
                        w = pw.BumpIsotropicWavelet(M, N, j, 0., k0=k0)
                        m = np.fft.ifft2(np.fft.fft2(nnpatch_)*np.fft.fft2(w.data)).real
                        nm_.append(m)
                        nm2.append(np.mean(m**2))
                    for j in range(j_min,J):
                            j1 = j
                        #for j1 in range(j_min,J):
                            nm3.append(np.mean(nm_[j-j_min]*nm_[j1-j_min]**2))
                    for j in range(j_min,J):
                        for j1 in range(j_min,J):
                            nm4.append(np.mean(nm_[j-j_min]*np.abs(nm_[j1-j_min])))
            
                    '''     
                    if (denoise1  != None):
                        s00 = np.mean(np.array(s00)-np.array(ns00),axis=0)
                        s11 = np.mean(np.array(s11)- np.array(ns11),axis=0)
                        s01 = np.mean(np.array(s01)-np.array(ns01),axis=0)
                        c01 = np.mean(np.array(c01)- np.array(nc01),axis=0)
                        cphase = np.mean(np.array(cphase)-np.array(ncphase),axis=0)
                        dv = np.hstack([s00,s11,s01,c01,cphase])
                    else:
                        s00 = np.mean(np.array(s00),axis=0)#-np.array(ns00)
                        s11 = np.mean(np.array(s11),axis=0)#- np.array(ns11)
                        s01 = np.mean(np.array(s01),axis=0)#-np.array(ns01)
                        c01 = np.mean(np.array(c01),axis=0)#- np.array(nc01)
                        cphase =  np.mean(np.array(cphase),axis=0)#-np.array(ncphase)
                        dv = np.hstack([s00,s11,s01,c01,cphase])      

                    #print (s00_indices,s11_indices,s01_indices,c01_indices,cphase_indices)
                    #j l p j l p
                    s00_indices =    np.array(['S00_'+'j'+str(s00_indices[ii][0])+'j'+str(s00_indices[ii][3]) for ii in range(len(s00_indices))])
                    s11_indices =    np.array(['S11_'+'j'+str(s11_indices[ii][0])+'j'+str(s11_indices[ii][3]) for ii in range(len(s11_indices))])
                    s01_indices =    np.array(['S01_'+'j'+str(s01_indices[ii][0])+'j'+str(s01_indices[ii][3]) for ii in range(len(s01_indices))])
                    c01_indices =    np.array(['C01_'+'j'+str(c01_indices[ii][0])+'j'+str(c01_indices[ii][3])+'dl'+str(c01_indices[ii][4]) for ii in range(len(c01_indices))])
                    cphase_indices = np.array(['Cphase_'+str(cphase_indices[ii][0])+str(cphase_indices[ii][3]) for ii in range(len(cphase_indices))])
    #

                    dv_indexes = np.hstack([s00_indices,s11_indices,s01_indices,c01_indices,cphase_indices])
                 
                    self.moments_pywph[label]['{0}_{1}'.format(i,j)] = copy.deepcopy(dv)
                    #if (i == 0) and( j==3):
                    #    print (dv.real[30])
                    try:
                        self.moments_pywph__[label]['{0}_{1}'.format(i,j)] = [copy.deepcopy(dv.real)]
                    except:
                        pass
                    self.moments_pywph_indexes[label]['{0}_{1}'.format(i,j)] = dv_indexes

                    #try:
                    #    print (k,(self.moments_pywph__[label]['{0}_{1}'.format(0,3)][0][30]))
                    #except:
                    #    pass
                    
                    
                    
                    
    def compute_moments_pywhm1(self,label,field1,field2,denoise1=None,denoise2=None):
        
        M = self.fields_patches[field1][list(self.fields_patches[field1].keys())[0]][0][0]
        N = self.fields_patches[field1][list(self.fields_patches[field1].keys())[0]][0][1]
        tomo_bins = list(self.fields_patches[field1].keys())

        J = self.conf['J']
        j_min = self.conf['j_min']
        L =  self.conf['L']
        dn = 0

        self.moments_pywph[label] = dict()
        try:
            self.moments_pywph__[label] = dict()
        except:
            pass
        self.moments_pywph_indexes[label] = dict()
        #if 1==1:
        #        i = 0
        #        j = 3
        bins_ =[]
        for i in tomo_bins:
            for j in tomo_bins:
                bins_.append([i,j])
                            
        patch1_ = copy.deepcopy(self.fields_patches[field1])
        patch2_ = copy.deepcopy(self.fields_patches[field2])
        if (denoise1  != None):
            npatch1_ = copy.deepcopy(self.fields_patches[denoise1])
            npatch2_ = copy.deepcopy(self.fields_patches[denoise2])
        else:
            npatch1_ = None
            npatch2_ = None
                             
                     
        # ---
        xlist = np.arange(len(bins_))
        agents = 5
        pool = multiprocessing.Pool(processes=agents)
        pairs = pool.map(partial(parallel_wph,  patch1_=patch1_,patch2_=patch2_,bins_=bins_,npatch1_=npatch1_,npatch2_=npatch2_,denoise1=denoise1
                     ,denoise2=denoise2,M=M,N=N,J=J,j_min=j_min,dn=dn,L=L), xlist)
        pool.close()
        pool.join()
                    
                        
                    

    def compute_phwmoments_sphere(self,label = '',field1 = 'kE',field2 = 'kE',tomo_bins=[0,1,2,3]):
        self.moments_pywph__[label] = dict()
        self.moments_pywph_indexes[label] = dict()
        J_min = self.conf['J_min']
        B = self.conf['B']
        N = self.conf['N']
        upsample = 1
        spin = 0

        L = self.conf['nside']*2


        J = pys2let_j_max(2, L, J_min)

        for ti in tomo_bins:
            for tj in tomo_bins:

                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)] = dict()

                self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)] = dict()

                # S00 *****
                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['S00'] = np.zeros((J-J_min))#['S00_'+'j'+str(j)+'j'+str(j)]
                self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['S00'] = []

                for jx,j in enumerate(range(J_min, J)):
                    self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['S00'].append('j'+str(j)+'j'+str(j))
                    for n in range(0, N):
                        self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['S00'][jx] += (1./N)*WPH_moments(self.smoothed_maps[field1][ti]['{0}_{1}'.format(j,n)],self.smoothed_maps[field2][tj]['{0}_{1}'.format(j,n)],0,0).real


                # S11 *****
                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['S11'] = np.zeros((J-J_min))#['S00_'+'j'+str(j)+'j'+str(j)]
                self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['S11'] = []

                for jx,j in enumerate(range(J_min, J)):
                    self. moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['S11'].append('j'+str(j)+'j'+str(j))
                    for n in range(0, N):
                        self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['S11'][jx] += (1./N)*WPH_moments(self.smoothed_maps[field1][ti]['{0}_{1}'.format(j,n)],self.smoothed_maps[field2][tj]['{0}_{1}'.format(j,n)],1,1).real


                # S01 *****
                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['S01'] = np.zeros((J-J_min))#['S00_'+'j'+str(j)+'j'+str(j)]
                self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['S01'] = []

                for jx,j in enumerate(range(J_min, J)):
                    self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['S01'].append('j'+str(j)+'j'+str(j))
                    for n in range(0, N):
                        self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['S01'][jx] += (1./N)*WPH_moments(self.smoothed_maps[field1][ti]['{0}_{1}'.format(j,n)],self.smoothed_maps[field2][tj]['{0}_{1}'.format(j,n)],0,1).real



                # C01 *****
                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['C01_dl0'] = np.zeros((J-J_min))#['S00_'+'j'+str(j)+'j'+str(j)]
                self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['C01_dl0'] = []

                for jx,j in enumerate(range(J_min, J)):
                    for jy,jj in enumerate(range(j+1, J)):
                        self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['C01_dl0'].append('j'+str(j)+'j'+str(jj))
                        for n in range(0, N):
                            self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['C01_dl0'][jx] += (1./N)*WPH_moments(self.smoothed_maps[field1][ti]['{0}_{1}'.format(j,n)],self.smoothed_maps[field2][tj]['{0}_{1}'.format(jj,n)],0,1).real


                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['C01_dl1'] =[]#['S00_'+'j'+str(j)+'j'+str(j)]
                self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['C01_dl1'] = []

                for jx,j in enumerate(range(J_min, J)):
                    for jy,jj in enumerate(range(j+1, J)):
                        self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['C01_dl1'].append('j'+str(j)+'j'+str(jj))
                        mute = 0.
                        for n in range(0, N):
                            if n == N-1:
                                mute += (1./N)*WPH_moments(self.smoothed_maps[field1][ti]['{0}_{1}'.format(j,n)],self.smoothed_maps[field2][tj]['{0}_{1}'.format(jj,0)],0,1).real                      
                            else:
                                mute += (1./N)*WPH_moments(self.smoothed_maps[field1][ti]['{0}_{1}'.format(j,n)],self.smoothed_maps[field2][tj]['{0}_{1}'.format(jj,n+1)],0,1).real
                        self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['C01_dl1'].append(mute)
                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['C01_dl1'] = np.array(self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['C01_dl1'])
                # CPHASE *****
                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['Cphase'] = []#['S00_'+'j'+str(j)+'j'+str(j)]
                self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['Cphase'] = []

                for jx,j in enumerate(range(J_min, J)):
                    for jy,jj in enumerate(range(j+1, J)):
                        self.moments_pywph_indexes[label]['{0}_{1}'.format(ti,tj)]['Cphase'].append('j'+str(j)+'j'+str(jj))
                        mute = 0.
                        for n in range(0, N):
                            mute += (1./N)*WPH_moments(self.smoothed_maps[field1][ti]['{0}_{1}'.format(j,n)],self.smoothed_maps[field2][tj]['{0}_{1}'.format(jj,n)],1,2**(jj-j)).real
                        self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['Cphase'].append(mute)
                self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['Cphase'] = np.array(self.moments_pywph__[label]['{0}_{1}'.format(ti,tj)]['Cphase'])


def phase_transform(field,p=0):
    return np.abs(field)*np.exp(1.j*p*np.angle(field))


def WPH_moments(field_1,field_2,p1,p2):
    f1 = phase_transform(field_1,p=p1)
    f2 = phase_transform(field_2,p=p2)
    # compute correlaiton:
    return np.mean(f1*np.conjugate(f2))-np.mean(f1)*np.mean(np.conjugate(f2))







import math
import frogress
import copy

def project_(i,pairs_,mask,pix_large_convert,map_test_uniform,xsize,res):
    ra_,dec_,i_ = pairs_[:,i]


    map_test_ = copy.deepcopy(mask)
    del mask
    
                  
    map_test_uniform_ = copy.deepcopy(map_test_uniform)
    del map_test_uniform
    mask_ = np.in1d(pix_large_convert,i_)
    del pix_large_convert
    map_test_[~mask_] = 0.
    map_test_uniform_[~mask_] = 0.
    base = np.sum(map_test_uniform_.flatten())
    del map_test_uniform_# more than 20% of the stamp!  
    del mask_
    if len(map_test_[map_test_!=0])>20:
        m_ref = hp.gnomview(map_test_, rot=(ra_,dec_), xsize=xsize ,no_plot=True,reso=res,return_projected_map=True)

   
        if (1.*np.sum((m_ref.data).flatten())/base)>0.2:
        
            del map_test_
            del m_ref
            
            # more than 20% of the stamp!
            gc.collect()
            return[ra_,dec_]
            #count_area += (res/60.)**2 *np.sum(m_ref)
     
    del map_test_
    gc.collect()
    
def project_2(i,pairs,xsize,field,res,pix_large_convert,nside_small):
    if  pairs[i]:
        
        #### version masked
        '''
        ra_,dec_ = pairs[i]
        ip = convert_to_pix_coord(ra_,dec_, nside=nside_small)
        field_ = copy.deepcopy(field)
        mask_ = np.in1d(pix_large_convert,ip)
        field_[~mask_] = 0.
    
        mt1 = hp.gnomview(field_, rot=(ra_,dec_), xsize=xsize ,no_plot=True,reso=res, return_projected_map=True)
        '''

        ra_,dec_ = pairs[i]
        mt1 = hp.gnomview(field, rot=(ra_,dec_), xsize=xsize ,no_plot=True,reso=res, return_projected_map=True)
        
        return mt1

    
    

            
            

    
    
def parallel_wph(i,patch1_,patch2_,bins_,npatch1_,npatch2_,denoise1,denoise2,M,N,J,j_min,dn,L):
    bin_i, bin_j = bins_[i]
    
    
    patch1_ = patch1_[bin_i]
    patch2_ = patch2_[bin_j]
    if (denoise1  != None):
        npatch1 = npatch1_[denoise1][bin_i]
        npatch2 = npatch2_[denoise2][bin_j]  
            
            
    wph_op = pw.WPHOp(M, N, J, L=L,j_min=j_min, dn=dn, device='cpu')
    wph = wph_op([patch1, patch2], cross=True, ret_wph_obj=True)
    
    wph.to_isopar()
    s00, s00_indices = wph.get_coeffs("S00")
    s11, s11_indices = wph.get_coeffs("S11")
    s01, s01_indices = wph.get_coeffs("S01")
    c01, c01_indices = wph.get_coeffs("C01")
    cphase, cphase_indices = wph.get_coeffs("Cphase")
    c00, c00_indices = wph.get_coeffs("C00")
    if (denoise1  != None):
        wph = wph_op([npatch1, npatch2], cross=True, ret_wph_obj=True)
        wph.to_isopar()
        ns00, s00_indices = wph.get_coeffs("S00")
        ns11, s11_indices = wph.get_coeffs("S11")
        ns01, s01_indices = wph.get_coeffs("S01")
        nc01, c01_indices = wph.get_coeffs("C01")
        ncphase, cphase_indices = wph.get_coeffs("Cphase")
        nc00, c00_indices = wph.get_coeffs("C00")



    if (denoise1  != None):
        s00 = np.mean(np.array(s00)-np.array(ns00),axis=0)
        s11 = np.mean(np.array(s11)- np.array(ns11),axis=0)
        s01 = np.mean(np.array(s01)-np.array(ns01),axis=0)
        c01 = np.mean(np.array(c01)- np.array(nc01),axis=0)
        cphase = np.mean(np.array(cphase)-np.array(ncphase),axis=0)
        dv = np.hstack([s00,s11,s01,c01,cphase])
    else:
        s00 = np.mean(np.array(s00),axis=0)#-np.array(ns00)
        s11 = np.mean(np.array(s11),axis=0)#- np.array(ns11)
        s01 = np.mean(np.array(s01),axis=0)#-np.array(ns01)
        c01 = np.mean(np.array(c01),axis=0)#- np.array(nc01)
        cphase =  np.mean(np.array(cphase),axis=0)#-np.array(ncphase)
        dv = np.hstack([s00,s11,s01,c01,cphase])      

    #print (s00_indices,s11_indices,s01_indices,c01_indices,cphase_indices)
    #j l p j l p
    s00_indices =    np.array(['S00_'+'j'+str(s00_indices[ii][0])+'j'+str(s00_indices[ii][3]) for ii in range(len(s00_indices))])
    s11_indices =    np.array(['S11_'+'j'+str(s11_indices[ii][0])+'j'+str(s11_indices[ii][3]) for ii in range(len(s11_indices))])
    s01_indices =    np.array(['S01_'+'j'+str(s01_indices[ii][0])+'j'+str(s01_indices[ii][3]) for ii in range(len(s01_indices))])
    c01_indices =    np.array(['C01_'+'j'+str(c01_indices[ii][0])+'j'+str(c01_indices[ii][3])+'dl'+str(c01_indices[ii][4]) for ii in range(len(c01_indices))])
    cphase_indices = np.array(['Cphase_'+str(cphase_indices[ii][0])+str(cphase_indices[ii][3]) for ii in range(len(cphase_indices))])
#

    dv_indexes = np.hstack([s00_indices,s11_indices,s01_indices,c01_indices,cphase_indices])
    

    return dv_indexes,copy.deepcopy(dv),copy.deepcopy(dv.real)
