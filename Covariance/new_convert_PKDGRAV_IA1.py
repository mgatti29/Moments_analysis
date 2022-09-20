#srun --nodes=4 --tasks-per-node=16 --cpus-per-task=4 --cpu-bind=cores  python new_convert_PKDGRAV_IA1.py


#srun --nodes=4 --tasks-per-node=64  python new_convert_PKDGRAV_IA4.py
#srun --nodes=4 --tasks-per-node=32 --cpus-per-task=2 --cpu-bind=cores  python new_convert_PKDGRAV_IA1.py



from Moments_analysis import moments_map
import pickle 
import healpy as hp
import numpy as np
import os
from astropy.table import Table
import gc
import pyfits as pf
from Moments_analysis import g2k_sphere
import timeit
import os
from bornraytrace import lensing as brk
import numpy as np
from bornraytrace import intrinsic_alignments as iaa
import bornraytrace
from astropy.table import Table    
import healpy as hp
import frogress
import pyfits as pf
from astropy.cosmology import z_at_value
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import cosmolopy.distance as cd
from scipy.interpolate import interp1d
import gc
import pandas as pd
import pickle
import multiprocessing
from functools import partial

def apply_random_rotation(e1_in, e2_in):
    np.random.seed() # CRITICAL in multiple processes !
    rot_angle = np.random.rand(len(e1_in))*2*np.pi #no need for 2?
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = + e1_in * cos + e2_in * sin
    e2_out = - e1_in * sin + e2_in * cos
    return e1_out, e2_out

def IndexToDeclRa(index, nside,nest= False):
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)

def convert_to_pix_coord(ra, dec, nside=1024):
    """
    Converts RA,DEC to hpix coordinates
    """

    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    return pix

def generate_randoms_radec(minra, maxra, mindec, maxdec, Ngen, raoffset=0):
    r = 1.0
    # this z is not redshift!
    zmin = r * np.sin(np.pi * mindec / 180.)
    zmax = r * np.sin(np.pi * maxdec / 180.)
    # parity transform from usual, but let's not worry about that
    phimin = np.pi / 180. * (minra - 180 + raoffset)
    phimax = np.pi / 180. * (maxra - 180 + raoffset)
    # generate ra and dec
    z_coord = np.random.uniform(zmin, zmax, Ngen)  # not redshift!
    phi = np.random.uniform(phimin, phimax, Ngen)
    dec_rad = np.arcsin(z_coord / r)
    # convert to ra and dec
    ra = phi * 180 / np.pi + 180 - raoffset
    dec = dec_rad * 180 / np.pi
    return ra, dec


def addSourceEllipticity(self,es,es_colnames=("e1","e2"),rs_correction=True,inplace=False):

		"""

		:param es: array of intrinsic ellipticities, 

		"""

		#Safety check
		assert len(self)==len(es)

		#Compute complex source ellipticity, shear
		es_c = np.array(es[es_colnames[0]]+es[es_colnames[1]]*1j)
		g = np.array(self["shear1"] + self["shear2"]*1j)

		#Shear the intrinsic ellipticity
		e = es_c + g
		if rs_correction:
			e /= (1 + g.conjugate()*es_c)

		#Return
		if inplace:
			self["shear1"] = e.real
			self["shear2"] = e.imag
		else:
			return (e.real,e.imag)

        
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)
        f.close()

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f)
        f.close()
    return mute



def gk_inv(K,KB,nside,lmax):

    alms = hp.map2alm(K, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsE = alms/( 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
   
    kalmsE[ell == 0] = 0.0

    
    alms = hp.map2alm(KB, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsB = alms/( 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
   
    kalmsB[ell == 0] = 0.0

    _,e1t,e2t = hp.alm2map([kalmsE,kalmsE,kalmsB] , nside=nside, lmax=lmax, pol=True)
    return e1t,e2t# ,r



def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048,nosh=True):
    """
    Convert shear to convergence on a sphere. In put are all healpix maps.
    """

    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!


    ell, emm = hp.Alm.getlm(lmax=lmax)
    if nosh:
        almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
        almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    else:
        almsE = alms[1] * 1.
        almsB = alms[2] * 1. 
    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0

    


    almssm = [alms[0], almsE, almsB]


    kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
    B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    return E_map, B_map, almsE



def rotate_map_approx(mask, rot_angles, flip=False,nside = 2048):
    alpha, delta = hp.pix2ang(nside, np.arange(len(mask)))

    rot = hp.rotator.Rotator(rot=rot_angles, deg=True)
    rot_alpha, rot_delta = rot(alpha, delta)
    if not flip:
        rot_i = hp.ang2pix(nside, rot_alpha, rot_delta)
    else:
        rot_i = hp.ang2pix(nside, np.pi-rot_alpha, rot_delta)
    rot_map = mask*0.
    rot_map[rot_i] =  mask[np.arange(len(mask))]
    return rot_map



    
        
def make_maps(seed):
    st = timeit.default_timer()
    
    # READ IN PARAMETERS ********************************************
    #seed = runstodo[run_count]
    f,p,params_dict = seed

    # SET COSMOLOGY ************************************************

                               
    config = dict()
    config['Om'] = params_dict['Omegam']
    config['sigma8'] =  params_dict['s8'] 
    config['ns'] =params_dict['ns']
    config['Ob'] = Ob
    config['h100'] = params_dict['h'] 
    config['num'] = params_dict['num'] 
    config['nside_out'] = 512
    config['nside'] = 512
    config['sources_bins'] = [0,1,2,3]
    config['dz_sources'] = [0.,0.,0.,0.]
    config['m_sources'] = [0.,0.,0.,0.]
    config['A_IA'] = params_dict['A'] 
    config['eta_IA'] = params_dict['E']
    config['z0_IA'] = 0.67
    config['2PT_FILE'] = '/global/homes/m/mgatti/Mass_Mapping/HOD/PKDGRAV_CODE//2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_6000HR.fits'  

    rot = params_dict['rot'] 

    

        
    cosmo1 = {'omega_M_0': config['Om'], 
     'omega_lambda_0':1-config['Om'],
     'omega_k_0':0.0, 
     'omega_b_0' : config['Ob'],
     'h':config['h100'],
     'sigma_8' : config['sigma8'],
     'n': config['ns']}

    cosmology = FlatLambdaCDM(H0= config['h100']*100. * u.km / u.s / u.Mpc, Om0=config['Om'])
    z_hr = np.linspace(0,10,5001)
    d_hr = cd.comoving_distance(z_hr,**cosmo1)*config['h100']

    f1 = interp1d(d_hr,z_hr)
    cosmo = FlatLambdaCDM(H0= config['h100']*100. * u.km / u.s / u.Mpc, Om0=config['Om'])

    
    # *******************
    # some stupid labels for intermediate products *******
    if params_dict['extralab']:
        ex_ = 'H0='+str(config['h100'])+'ns='+str(config['ns'])+'Ob='+str(config['Ob'])
    else:
        ex_=''
                
            
            
    if params_dict['SC']:
        label_output0 = 'NEW_grid_fiducialcosmo_SC_k_'+ex_+'Om='+str(config['Om'])+'_s8='+str(config['sigma8'])+'num='+str(config['num'])
    else:
        label_output0 = 'NEW_grid_fiducialcosmo_k_'+ex_+'Om='+str(config['Om'])+'_s8='+str(config['sigma8'])+'num='+str(config['num'])
        

        
    # SAVE LENS PLANES ********************************************************
    if params_dict['extralab']:
        shell_directory = path_sims+'/cosmo_H0={0}_Ob={1}_Om={2}_mn=0.02_ns={3}_num={4}_s8={5}/'.format(config['h100'],config['Ob'],config['Om'],config['ns'],config['num'],config['sigma8'])
    else:
        shell_directory = path_sims+'/cosmo_Om={1}_num={0}_s8={2}/'.format(config['num'],config['Om'],config['sigma8'])
        
        
        
        
    from ekit import paths as path_tools
                                                                                                       
    #print ('shell_directory: ',shell_directory)
    shell_files = glob.glob("{}DES-Y3-shell*".format(shell_directory))
    z_bounds, _ = path_tools.get_parameters_from_path(shell_files)
    shell_files = np.sort(shell_files)

    z_bounds['z-high'] = np.sort(z_bounds['z-high'])
    z_bounds['z-low'] = np.sort(z_bounds['z-low'])
    z_bin_edges = np.hstack([z_bounds['z-low'],z_bounds['z-high'][-1]])

    for s_ in (range(len(z_bounds['z-high']))):
        path_ = output_base+'/'+label_output0+'/lens_{0}_{1}.fits'.format(s_,config['nside_out'])
        if not os.path.exists(path_):
            shell_ = hp.read_map(shell_files[s_])
            shell_ =  (shell_-np.mean(shell_))/np.mean(shell_)
            shell_ = hp.ud_grade(shell_, nside_out = config['nside_out'])


                                                                                                       

                    #m_4 = hp.alm2map(hp.map2alm(m_4,lmax = 2048*2),nside = config['nside_out'])
            fits_f = Table()
            fits_f['T'] = shell_
            if os.path.exists(path_):
                os.remove(path_)

            fits_f.write(path_)

    # SAVE CONVERGENCE PLANES ********************************************************
    kappa_pref_evaluated = brk.kappa_prefactor(cosmology.H0, cosmology.Om0, length_unit = 'Mpc')
    comoving_edges = [cosmology.comoving_distance(x_) for x_ in np.array((z_bounds['z-low']))]


    z_centre = np.empty((len(comoving_edges)-1))
    for i in range(len(comoving_edges)-1):
        z_centre[i] = z_at_value(cosmology.comoving_distance,0.5*(comoving_edges[i]+comoving_edges[i+1]))

    un_ = comoving_edges[:(i+1)][0].unit
    comoving_edges = np.array([c.value for c in comoving_edges])
    comoving_edges = comoving_edges*un_


    overdensity_array = [np.zeros(hp.nside2npix(config['nside_out']))]


    path_ = output_base+'/'+label_output0+'/gg_{0}_{1}.fits'.format(len(z_bounds['z-high'])-1,config['nside_out'])


    if not os.path.exists(path_):

        #print ('load lens')
        for s_ in (range(len(z_bounds['z-high']))):
            try:
                path_ = output_base+'/'+label_output0+'/lens_{0}_{1}.fits'.format(s_,config['nside_out'])
                m_ = pf.open(path_)
                overdensity_array.append(m_[1].data['T'])
            except:
                if shell !=0:
                    overdensity_array.append(np.zeros(hp.nside2npix(config['nside_out'])))
                #pass

        #print ('done ++++')
        overdensity_array = np.array(overdensity_array)
        #print(overdensity_array.shape)

        from bornraytrace import lensing
        kappa_lensing = np.copy(overdensity_array)*0.


        #comoving_edges = np.array([c.value for c in comoving_edges])
       # print ('doing kappa')
        for i in (np.arange(kappa_lensing.shape[0])):
            try:
                kappa_lensing[i] = lensing.raytrace(cosmology.H0, cosmology.Om0,
                                             overdensity_array=overdensity_array[:i].T,
                                             a_centre=1./(1.+z_centre[:i]),
                                             comoving_edges=comoving_edges[:(i+1)])
            except:
                pass
        #print ('done kappa')
        for s_ in range(kappa_lensing.shape[0]):
            #try:
            path_ = output_base+'/'+label_output0+'/kappa_{0}_{1}.fits'.format(s_,config['nside_out'])
            #if not os.path.exists(path_):
            #    
            ##    if os.path.exists(path_):
            ##        os.remove(path_)
#
            #    fits_f = Table()
            #    fits_f['T'] = kappa_lensing[s_]
            #    fits_f.write(path_)
            ##except:
            ##    pass
        #print ('save kappa done')

        # mage g1 and g2 ---
        for i in (range(kappa_lensing.shape[0])):
            path_ = output_base+'/'+label_output0+'/gg_{0}_{1}.fits'.format(i,config['nside_out'])

            if not os.path.exists(path_):

                g1_, g2_ = gk_inv(kappa_lensing[i]-np.mean(kappa_lensing[i]),kappa_lensing[i]*0.,config['nside_out'],config['nside_out']*2)
                g1_IA, g2_IA = gk_inv(overdensity_array[i]-np.mean(overdensity_array[i]),kappa_lensing[i]*0.,config['nside_out'],config['nside_out']*2)


                fits_f = Table()
                fits_f['g1'] = g1_
                fits_f['g2'] = g2_
                fits_f['g1_IA'] = g1_IA
                fits_f['g2_IA'] = g2_IA
                fits_f.write(path_)




    c1 = (5e-14 * (u.Mpc**3.)/(u.solMass * u.littleh**2) ) 
    c1_cgs = (c1* ((u.littleh/(cosmology.H0.value/100))**2.)).cgs
    rho_c1 = (c1_cgs*cosmology.critical_density(0)).value



    try:
        del overdensity_array
        del kappa_lensing
        gc.collect()
    except:
        pass

    mu = pf.open(config['2PT_FILE'])



    redshift_distributions_sources = {'z':None,'bins':dict()}
    redshift_distributions_sources['z'] = mu[6].data['Z_MID']
    for ix in config['sources_bins']:
        redshift_distributions_sources['bins'][ix] = mu[6].data['BIN{0}'.format(ix+1)]
    mu = None




    g1_tomo = dict()
    g2_tomo = dict()
    d_tomo = dict()
    nz_kernel_sample_dict = dict()
    for tomo_bin in config['sources_bins']:
        g1_tomo[tomo_bin] = np.zeros(hp.nside2npix(config['nside']))
        g2_tomo[tomo_bin] = np.zeros(hp.nside2npix(config['nside']))
        d_tomo[tomo_bin] = np.zeros(hp.nside2npix(config['nside']))
        redshift_distributions_sources['bins'][tomo_bin][250:] = 0.
        nz_sample = brk.recentre_nz(np.array(z_bin_edges).astype('float'),  redshift_distributions_sources['z']+config['dz_sources'][tomo_bin],  redshift_distributions_sources['bins'][tomo_bin] )
        nz_kernel_sample_dict[tomo_bin] = nz_sample*(z_bin_edges[1:]-z_bin_edges[:-1])


    for i in (range(2,len(comoving_edges)-1)):

            path_ = output_base+'/'+label_output0+'/lens_{0}_{1}.fits'.format(i,config['nside_out'])
            #pathk_ = output+'/'+label_output1+'/kappa_{0}_{1}.fits'.format(i-2,config['nside_out'])

            pathgg_ = output_base+'/'+label_output0+'/gg_{0}_{1}.fits'.format(i,config['nside_out'])

            #fits_f = Table()
            #fits_f['g1'] = g1_
            #fits_f['g2'] = g2_
            #fits_f['g1_IA'] = g1_IA
            #fits_f['g2_IA'] = g2_IA
            #fits_f.write(path_)
           # try:
            if 1==1:
                k_ = pf.open(pathgg_)
                d_ = pf.open(path_)
                IA_f = iaa.F_nla(z_centre[i], cosmology.Om0, rho_c1=rho_c1,A_ia = config['A_IA'], eta=config['eta_IA'], z0=config['z0_IA'],  lbar=0., l0=1e-9, beta=0.)
                #print ((k_[1].data['T']))
                for tomo_bin in config['sources_bins']:         
                    m_ = 1.#+config['m_sources'][tomo_bin-1]
                    b = 1.#extra_params['db'][tomo_bin]
                    if SC:
                        g1_tomo[tomo_bin]  +=  m_*((1.+(b*d_[1].data['T']))*(k_[1].data['g1']+k_[1].data['g1_IA']*IA_f))*nz_kernel_sample_dict[tomo_bin][i]
                        g2_tomo[tomo_bin]  +=  m_*((1.+(b*d_[1].data['T']))*(k_[1].data['g2']+k_[1].data['g2_IA']*IA_f))*nz_kernel_sample_dict[tomo_bin][i]
                        d_tomo[tomo_bin] +=  (1.+b*d_[1].data['T'])*nz_kernel_sample_dict[tomo_bin][i]
                    else:
                        g1_tomo[tomo_bin]  +=  m_*(k_[1].data['g1']+k_[1].data['g1_IA']*IA_f)*nz_kernel_sample_dict[tomo_bin][i]
                        g2_tomo[tomo_bin]  +=  m_*(k_[1].data['g2']+k_[1].data['g2_IA']*IA_f)*nz_kernel_sample_dict[tomo_bin][i]
                        d_tomo[tomo_bin] +=  (1.+d_[1].data['T'])*nz_kernel_sample_dict[tomo_bin][i]



        

    '''
    should save to temp_ maps_.
    should also load to memory the catalog per bin **
    '''

    #config['nside'] = 512
    #for tomo_bin in config['sources_bins']:  
    #    g1_tomo[tomo_bin] = hp.ud_grade(g1_tomo[tomo_bin],nside_out = config['nside'] )
    #    g2_tomo[tomo_bin] = hp.ud_grade(g2_tomo[tomo_bin],nside_out = config['nside'] )
    #    d_tomo[tomo_bin] = hp.ud_grade(d_tomo[tomo_bin],nside_out = config['nside'] )
    #    
        
    
    # need to rotate the maps first -----------------------------------------------------------    
    #print ('')
    #print ('ROTATING MAPS ++++++++')
    for i in config['sources_bins']:
        if rot ==0:
            pass
        elif (rot ==1):
            g1_tomo[tomo_bin] = rotate_map_approx(g1_tomo[tomo_bin],[ 180 ,0 , 0], flip=False,nside = config['nside'])
            g2_tomo[tomo_bin] = rotate_map_approx(g2_tomo[tomo_bin],[ 180 ,0 , 0], flip=False,nside = config['nside'])
            d_tomo[tomo_bin] = rotate_map_approx(d_tomo[tomo_bin],[ 180 ,0 , 0], flip=False,nside = config['nside'])
        elif rot ==2:
            g1_tomo[tomo_bin] = rotate_map_approx(g1_tomo[tomo_bin],[ 90 ,0 , 0], flip=True,nside = config['nside'])
            g2_tomo[tomo_bin] = rotate_map_approx(g2_tomo[tomo_bin],[ 90 ,0 , 0], flip=True,nside = config['nside'])
            d_tomo[tomo_bin] = rotate_map_approx(d_tomo[tomo_bin],[ 90 ,0 , 0], flip=True,nside = config['nside'])
        elif rot ==3:
            g1_tomo[tomo_bin] = rotate_map_approx(g1_tomo[tomo_bin],[ 270 ,0 , 0], flip=True,nside = config['nside'])
            g2_tomo[tomo_bin] = rotate_map_approx(g2_tomo[tomo_bin],[ 270 ,0 , 0], flip=True,nside = config['nside'])
            d_tomo[tomo_bin] = rotate_map_approx(d_tomo[tomo_bin],[ 270 ,0 , 0], flip=True,nside = config['nside'])


   # print ('LOADING THE CATALOGS MAPS ++++++++')
    st1 = timeit.default_timer()



    maps_PKDGRAV = dict()

    sources_maps = dict()
    for noise_rel in range(noise_rels):
        sources_maps[noise_rel] = dict()
    #print ('doing the mpas!')
  
        
    for tomo_bin in config['sources_bins']:   
   
        mcal_catalog = load_obj('/global/cfs/cdirs/des/mass_maps/Maps_final/data_catalogs_weighted_{0}'.format(tomo_bin))
    
    
        maps_PKDGRAV[tomo_bin] = dict()
        dec1 = mcal_catalog[tomo_bin]['dec']
        ra1 = mcal_catalog[tomo_bin]['ra']
        w = mcal_catalog[tomo_bin]['w'] 
        pix = convert_to_pix_coord(ra1,dec1, nside=config['nside'])
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)

        g1_ = g1_tomo[tomo_bin][pix]
        g2_ = g2_tomo[tomo_bin][pix]
            
        if SC:
            f = 1./np.sqrt(d_tomo[tomo_bin]/np.sum(nz_kernel_sample_dict[tomo_bin]))
            f = f[pix]
        else:
            f = 1.
            
        
        if 1==1:
            #print ('noise_rel: ',noise_rel)
            pp_temp = output_temp+p+'_'+str(tomo_bin)
            pp_ = output+p
           # if 1==1:
            #if not os.path.exists(pp_temp+'.pkl'):
            if not os.path.exists(pp_+'.pkl'):
                    sources_maps[noise_rel][tomo_bin] = dict()
                    n_map = np.zeros(hp.nside2npix(config['nside']))
                    n_map_sc = np.zeros(hp.nside2npix(config['nside']))


                    n_map[unique_pix] += np.bincount(idx_rep, weights=w)
                    n_map_sc[unique_pix] += np.bincount(idx_rep, weights=w/f**2)



                    es1,es2 = apply_random_rotation(mcal_catalog[tomo_bin]['e1']/f, mcal_catalog[tomo_bin]['e2']/f)
                    es1a,es2a = apply_random_rotation(mcal_catalog[tomo_bin]['e1']/f, mcal_catalog[tomo_bin]['e2']/f)

                    x1_sc,x2_sc = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1,'e2':es2},es_colnames=("e1","e2"))

                    e1_map_buzz = np.zeros(hp.nside2npix(config['nside']))
                    e2_map_buzz = np.zeros(hp.nside2npix(config['nside']))
                    e1r_map_buzz = np.zeros(hp.nside2npix(config['nside']))
                    e2r_map_buzz = np.zeros(hp.nside2npix(config['nside']))


                    e1_map_buzz[unique_pix] += np.bincount(idx_rep, weights= x1_sc*w)
                    e2_map_buzz[unique_pix] += np.bincount(idx_rep, weights= x2_sc*w)
                    e1r_map_buzz[unique_pix] += np.bincount(idx_rep, weights=es1a*w)
                    e2r_map_buzz[unique_pix] += np.bincount(idx_rep, weights=es2a*w)

                    mask_sims = n_map_sc != 0.
                    e1_map_buzz[mask_sims]  = e1_map_buzz[mask_sims]/(n_map_sc[mask_sims])
                    e2_map_buzz[mask_sims] =  e2_map_buzz[mask_sims]/(n_map_sc[mask_sims])
                    e1r_map_buzz[mask_sims]  = e1r_map_buzz[mask_sims]/(n_map_sc[mask_sims])
                    e2r_map_buzz[mask_sims] =  e2r_map_buzz[mask_sims]/(n_map_sc[mask_sims])


                    EE,BB,_   =  g2k_sphere(e1_map_buzz, e2_map_buzz, mask_sims, nside=config['nside'], lmax=config['nside']*2 ,nosh=True)
                    EEn,BBn,_ =  g2k_sphere(e1r_map_buzz, e2r_map_buzz, mask_sims, nside=config['nside'], lmax=config['nside']*2 ,nosh=True)

                    EE[~mask_sims] = 0.
                    EEn[~mask_sims] = 0.
                    BB[~mask_sims] = 0.
                    BBn[~mask_sims] = 0.

                    #sources_maps[tomo_bin] = {'g1_map':g1_map,'g2_map':g2_map,'EE':EE,'EEn':EEn,'BB':BB,'BBn':BBn,'mask':mask_sims} 
                    dict_temp = {'EE':EE,'EEn':EEn,'BB':BB,'BBn':BBn,'mask':mask_sims} 
                    save_obj(pp_temp,dict_temp)
                    sources_maps[noise_rel][tomo_bin] = None #{'EE':EE,'EEn':EEn,'BB':BB,'BBn':BBn,'mask':mask_sims} 

        del mcal_catalog[tomo_bin]
        gc.collect()
            

    
    def compute_phmoments(sources_maps = None,output='',output_temp='',params_dict=''):
      
  
        
        #try:
            print ('SAVING ',output)
            if not os.path.exists(output+'.pkl'):

                conf = dict()
                conf['j_min'] = 0
                conf['J'] = 5
                conf['B'] = 2
                conf['L'] = 2
                conf['nside'] =512
                conf['lmax'] = conf['nside']*2
                conf['verbose'] = False
                conf['output_folder'] = output+'/test/'


                mcal_moments = moments_map(conf)
                #mask_DES_y3 = load_obj('/global/cfs/cdirs/des//mass_maps/Maps_final//mask_DES_y3')

                #mask_DES_y3
                # this add the maps
                tomo_bins = [0,1,2,3]
                for t in tomo_bins:
                    pp_temp = output_temp+'_'+str(t)
                    dict_temp = load_obj(pp_temp)

                    mcal_moments.add_map(dict_temp['EE'], field_label = 'k', tomo_bin = t)
                    mcal_moments.add_map(dict_temp['EEn'], field_label = 'kn', tomo_bin = t)
                    if t == 3:
                        mcal_moments.mask = dict_temp['mask'] 
                    os.remove(pp_temp +'.pkl')
                del  dict_temp
                gc.collect()
                
       
                mcal_moments.cut_patches( nside=512, nside_small=16)
                mcal_moments.moments_pywph = dict()
                mcal_moments.moments_pywph_indexes = dict()


                
                # maybe we can parallelise this ----

                print ('compute moments')
                mcal_moments.compute_moments_pywhm(label = 'NK',field1='kn',field2='k')
                print ('KN')
                mcal_moments.compute_moments_pywhm(label = 'KN',field1='k',field2='kn')
                print ('NN')
                mcal_moments.compute_moments_pywhm(label = 'NN',field1='kn',field2='kn')
                print ('KK')
                mcal_moments.compute_moments_pywhm(label = 'KK',field1='k',field2='k')




                try:
                    del mcal_moments.fields
                    del mcal_moments.fields_patches
                    #del mcal_moments.smoothed_maps
                    gc.collect()
                except:
                    pass
                #print ('save')


                save_obj(output,[mcal_moments,params_dict])
        #except:
        #        #pp_temp = output+'/'+label_output0+'/temp/{0}_{1}_{2}'.format(noise_rel,0,rot+1)
        #            
        #        print ('FAILED ',output)
#
                
                
    compute_phmoments(sources_maps,output+p,output_temp+p,params_dict)
   # st3 = timeit.default_timer()
   # print ('PRE WORK TIME ', st2-st)
    #for x_ in xlist:
    #    compute_phmoments(x_, sources_maps = sources_maps,output=output,label_output0=label_output0,label_output1=label_output1,num=num,rot=rot)
   # st3 = timeit.default_timer()
   # print ('AFTER WORK TIME ', st3-st2)
    #_ = pool.map(partial(compute_phmoments, sources_maps = sources_maps,output=output,label_output0=label_output0,label_output1=label_output1,num=num,rot=rot), xlist)
           
        
    

 
   
    

# NSIDE OF THE MAPS YOU'LL BE CREATING **********************
nside = 512

SC = True

no_IA = False
IA_file = 'params_IA_lin_1'
noise_rels = 10
rot_num = 4




# CHANGE THIS TO A FOLDER IN YOUR CSCRATCH1 *****************

# CHANGE THIS TO A FOLDER IN YOUR CSCRATCH1 *****************
output_base = '/pscratch/sd/m/mgatti/DarkGrid///'
output = '/global/cfs/cdirs/des/mgatti/DarkGrid//runs_y3_IA_512_newtests_eta/'
output_temp = '/pscratch/sd/m/mgatti/DarkGrid//temp_stuff1/'

# path to sims
# path to sims
#path_sims = '/global/cfs/cdirs/des/darkgrid/fidu_run_1/'
#path_sims = '/global/cfs/cdirs/des/darkgrid/fidu_delta_run_5/'
path_sims = '/global/cfs/cdirs/des/darkgrid/grid_run_1/'
#path_sims = '/global/cscratch1/sd/dominikz/DES_Y3_PKDGRAV_SIMS/grid_run_1/'



if not os.path.exists(output_temp):
    try:
        os.mkdir(output_temp)
    except:
        pass

  



        
    


if __name__ == '__main__':
    
    
 #for h__ in range(1,8):
 #   IA_file = 'params_IA_lin_{0}'.format(h__)

    import glob
    
    f_ = glob.glob(path_sims+'/cosmo_*')
    runstodo=[]
    count = 0
    miss = 0

    
    
    print ('COSMOLOGIES ', len(f_))
    
    if no_IA:
        ia_rel=1
    else:
        m = np.load('./{0}.npy'.format(IA_file),allow_pickle=True).item()
        ia_rel = 8 #m['rel']
        
    print ('IA rel ', ia_rel)
       
        
    for f in f_:
        
        # requirements su Om & s8 to restrict to fiducial 50.
        Omegam = f.split('Om=')[1].split('_')[0]
        s8 = f.split('s8=')[1].split('_')[0]
        num = f.split('num=')[1].split('_')[0]
        
                            
        #if (np.float(Omegam)==0.26) & (np.int(num)<5) &  (np.float(s8)==0.84):
        if 1==1:                                                                                            
            try:

                ns = np.float(f.split('ns=')[1].split('_')[0])
                Ob =  np.float(f.split('Ob=')[1].split('_')[0])
                h =  np.float(f.split('H0=')[1].split('_')[0])

                extralab = True

            except:
                extralab = False
                # default values ++++
                ns = 0.96
                Ob = 0.048
                h = 0.673

                # make label **************************
            ex_ = '_H0='+str(h)+'_ns='+str(ns)+'_Ob='+str(Ob)
            
                
            if SC:
                ex_ = '_SC_'+ex_

        
            for ia_i in range(ia_rel):
                if no_IA:
                        A_IA = 0.
                        e_IA = 0.
                else:

                    p = '{0}_{1}'.format(Omegam,s8)
                    A_IA = m['A'][m['om_s8'] == p,ia_i][0]
                    e_IA = m['E'][m['om_s8'] == p,ia_i][0]




                 #4x
                for i in range(rot_num):
                    
                    for nn in range(noise_rels):
                        
                        params_dict = dict()
                        params_dict['Omegam'] = np.float(Omegam)
                        params_dict['s8'] = np.float(s8)
                        params_dict['num'] = np.float(num)
                        params_dict['A'] = A_IA
                        params_dict['E'] = e_IA
                        params_dict['noise'] = nn
                        params_dict['num'] = num
                        
                        params_dict['rot'] = i
                        params_dict['ns'] = np.float(ns)
                        params_dict['h'] = np.float(h)
                        params_dict['ob'] = np.float(Ob)
                        params_dict['extralab'] = extralab
                        params_dict['SC'] = SC
                        
            
                        p = '_Om='+str(Omegam)+'_s8='+str(s8)+'_num='+str(num)+ex_+'_A_IA={0:2.4f}'.format(A_IA)+'_e_IA={0:2.4f}'.format(e_IA)+'_'+str(i+1)+'_noise_'+str(nn)

                        #print (p,params_dict)
                        
                        
                        if not os.path.exists(output+p+'.pkl'):
                            runstodo.append([f,p,params_dict])
                            miss+=1
                        else:
                            count +=1
                   
                        
                    

                            
                            
            


           
    
    
    
    print (len(runstodo),count,miss)

   # make_maps(runstodo[run_count])
    run_count=0
    from mpi4py import MPI 
    while run_count<len(runstodo):
        comm = MPI.COMM_WORLD
        #print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        #try:
        #if comm.rank == 2:
        if (run_count+comm.rank)<len(runstodo):
      #          print (run_count,comm.rank,run_count+comm.rank,runstodo[run_count+comm.rank])
                make_maps(runstodo[run_count+comm.rank])
    #    except:
     #       pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
####
#