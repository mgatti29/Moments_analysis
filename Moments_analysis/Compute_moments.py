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
from .smoothing_utils import almxfl
from astropy.table import Table

class moments_map(object):
    def __init__(self,conf = {'output_folder': './'}):
        '''
        Initialise the moments_map object.
        This object contains: 
        1) configurations
        2) fileds -> it's a dictionary with the input maps
        3) smoothed_maps: those are the smoothed version of the input maps. So far we have implemented top-hat smoothing.
        4) moments: 2nd and 3rd moments of the smoothed maps.
        
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
                    else:
                        if not skip_loading_smoothed_maps:
                            # load the map
                            mute = pf.open(path,memmap=False)
                            mapaa = copy.copy(mute[1].data['map'])
                        else:
                            mapaa = None
                    if not skip_loading_smoothed_maps:
                        self.smoothed_maps[output_label+'_'+ll[ix]][binx][sm] = copy.deepcopy(mapaa)
                        del mapaa
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
                        map_signal1 = self.smoothed_maps[field_label1][bin1][sm]
                        map_signal2 = self.smoothed_maps[field_label2][bin2][sm]

                        if (denoise1  != None) and (i==j) and (denoise1 == denoise2):
                            map_rndm1 = self.smoothed_maps[denoise1][bin1][sm]
                            map_rndm2 = self.smoothed_maps[denoise2][bin2][sm]                        
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
                            map_tbsm1 = self.smoothed_maps[field_label1][bin1][sm]
                            map_tbsm2 = self.smoothed_maps[field_label2][bin2][sm]
                            map_tbsm3 = self.smoothed_maps[field_label2][bin3][sm]
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
