#srun --nodes=4 --tasks-per-node=8 --cpus-per-task=8 --cpu-bind=cores  python new_convert_PKDGRAV_random.py
#srun --nodes=4 --tasks-per-node=20 --cpus-per-task=3 --cpu-bind=cores  python new_convert_PKDGRAV_random.py


#srun --nodes=4 --tasks-per-node=2 --cpus-per-task=32 --cpu-bind=cores  python new_convert_PKDGRAV_random.py

import pandas as pd

from Moments_analysis import moments_map
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
import pandas as pdf
import pickle

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


def addSourceEllipticity(self,es,es_colnames=("e1","e2"),rs_correction=False,inplace=False):

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



def random_draw_ell_from_w(wi,w,e1,e2):
    '''
    wi: input weights
    w,e1,e2: all the weights and galaxy ellipticities of the catalog.
    e1_,e2_: output ellipticities drawn from w,e1,e2.
    '''


    ell_cont = dict()
    for w_ in np.unique(w):
        mask_ = w == w_
        w__ = np.int(w_*10000)
        ell_cont[w__] = [e1[mask_],e2[mask_]]

    e1_ = np.zeros(len(wi))
    e2_ = np.zeros(len(wi))


    for w_ in np.unique(wi):
        mask_ = (wi*10000).astype(np.int) == np.int(w_*10000)
        e1_[mask_] = ell_cont[np.int(w_*10000)][0][np.random.randint(0,len(ell_cont[np.int(w_*10000)][0]),len(e1_[mask_]))]
        e2_[mask_] = ell_cont[np.int(w_*10000)][1][np.random.randint(0,len(ell_cont[np.int(w_*10000)][0]),len(e1_[mask_]))]

    return e1_,e2_



def make_maps(seed):

    
    # READ IN PARAMETERS ********************************************
    #seed = runstodo[run_count]
    f,rot,nr = seed
    
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

    
    Omegam = np.float(f.split('Om=')[1].split('_')[0])
    s8 =  np.float(f.split('s8=')[1].split('_')[0])
    num = f.split('num=')[1].split('_')[0]
    
    print ('RUN -', Omegam,s8,num,rot,SC)

    # SET COSMOLOGY ************************************************

    config = dict()
    config['Om'] = Omegam#0.3071
    config['sigma8'] =  s8#0.8228
    config['ns'] =ns
    config['Ob'] = Ob
    config['h100'] = h
    config['nside_out'] = 1024
    config['nside'] = 1024
    config['sources_bins'] = [0,1,2,3]
    config['dz_sources'] = [0.,0.,0.,0.]
    config['m_sources'] = [0.,0.,0.,0.]
    config['A_IA'] = 0.
    config['eta_IA'] = 0.
    config['z0_IA'] = 0.67
    config['2PT_FILE'] = '/global/homes/m/mgatti/Mass_Mapping/HOD/PKDGRAV_CODE//2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_6000HR.fits'  



        
    try:
        m = np.load('./params_IA.npy',allow_pickle=True).item()
        p = '{0}_{1}'.format(Omegam,s8)
        A_IA = m['A'][m['om_s8'] == p]
        e_IA = m['E'][m['om_s8'] == p]
        if len(A_IA)==0:
            A_IA = 0.
            e_IA = 0.
        else:
            A_IA = A_IA[0]
            e_IA = e_IA[0]
    except:
        A_IA = 0.
        e_IA = 0.

    print ('IA---- : ',A_IA,e_IA)

    config['A_IA'] = A_IA
    config['eta_IA'] = e_IA
    
    # make label **************************
    if extralab:
        ex_ = 'H0='+str(h)+'ns='+str(ns)+'Ob='+str(Ob)
    else:
        ex_=''
   
        
    
        
        
    label_output0 = label_output+ex_+'Om='+str(Omegam)+'_s8='+str(s8)+'num='+str(num)
    
    
    print (output+'/'+label_output0)
    
    ex_ = 'RANDOM_'+ex_
        
    label_output1 = label_output+ex_+'Om='+str(Omegam)+'_s8='+str(s8)+'num='+str(num)+'A_IA={0:2.2f}'.format(A_IA)+'e_IA={0:2.2f}'.format(e_IA)
    #label_output1_ = label_output_+ex_+'Om='+str(Omegam)+'_s8='+str(s8)+'num='+str(num)+'A_IA={0:2.2f}'.format(A_IA)+'e_IA={0:2.2f}'.format(e_IA)
    if SC:
        print ('DOING ',output+'/runs_y3/'+label_output1+'_'+str(rot+1))
   # else:
   #     print ('DOING ',output+'/runs_y3/'+label_output1_+'_'+str(rot+1))
   #     
    print (output+'/'+label_output0+'/')
    
    if not os.path.exists(output+'/'+label_output0+'/'):
        try:
            os.mkdir(output+'/'+label_output0+'/')
        except:
            pass
        
  
        
    cosmo1 = {'omega_M_0': config['Om'], 
     'omega_lambda_0':1-config['Om'],
     'omega_k_0':0.0, 
     'omega_b_0' : config['Ob'],
     'h':config['h100'],
     'sigma_8' : config['sigma8'],
     'n': config['ns']}


    # ****************************************************************************************************************
    # THE FOLLOWING PARTS READ DARKGRID RAW PARTICLES, MAKE LENS & SHEAR PLANES
    cosmology = FlatLambdaCDM(H0= config['h100']*100. * u.km / u.s / u.Mpc, Om0=config['Om'])
    z_hr = np.linspace(0,10,5001)
    d_hr = cd.comoving_distance(z_hr,**cosmo1)*config['h100']

    f1 = interp1d(d_hr,z_hr)
    cosmo = FlatLambdaCDM(H0= config['h100']*100. * u.km / u.s / u.Mpc, Om0=config['Om'])


    # SAVE LENS PLANES ********************************************************
    if extralab:
        
        shell_directory = path_sims+'/cosmo_H0={0}_Ob={1}_Om={2}_mn=0.02_ns={3}_num={4}_s8={5}/'.format(h,Ob,Omegam,ns,num,s8)
    else:
        shell_directory = path_sims+'/cosmo_Om={1}_num={0}_s8={2}/'.format(num,Omegam,s8)
    from ekit import paths as path_tools
                                                                                                       
    print ('shell_directory: ',shell_directory)
    shell_files = glob.glob("{}DES-Y3-shell*".format(shell_directory))
    z_bounds, _ = path_tools.get_parameters_from_path(shell_files)
    shell_files = np.sort(shell_files)

    z_bounds['z-high'] = np.sort(z_bounds['z-high'])
    z_bounds['z-low'] = np.sort(z_bounds['z-low'])
    z_bin_edges = np.hstack([z_bounds['z-low'],z_bounds['z-high'][-1]])

    for s_ in frogress.bar(range(len(z_bounds['z-high']))):
        path_ = output+'/'+label_output0+'/lens_{0}_{1}.fits'.format(s_,config['nside_out'])
        if not os.path.exists(path_):
            shell_ = hp.read_map(shell_files[s_])
            shell_ =  (shell_-np.mean(shell_))/np.mean(shell_)
            shell_ = hp.ud_grade(shell_, nside_out = config['nside_out'])
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


    path_ = output+'/'+label_output0+'/gg_{0}_{1}.fits'.format(len(z_bounds['z-high'])-1,config['nside_out'])

    


    if not os.path.exists(path_):

        print ('load lens')
        for s_ in frogress.bar(range(len(z_bounds['z-high']))):
            
            try:
        
                    path_ = output+'/'+label_output0+'/lens_{0}_{1}.fits'.format(s_,config['nside_out'])
                    m_ = pf.open(path_)
                    overdensity_array.append(m_[1].data['T'])
            
                    
            except:
                if s_ !=0:
                    overdensity_array.append(np.zeros(hp.nside2npix(config['nside_out'])))
                #pass

        print ('done ++++')
        overdensity_array = np.array(overdensity_array)
        print(overdensity_array.shape)

        from bornraytrace import lensing
        kappa_lensing = np.copy(overdensity_array)*0.


        #comoving_edges = np.array([c.value for c in comoving_edges])
        print ('doing kappa')
        for i in frogress.bar(np.arange(kappa_lensing.shape[0])):
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
            path_ = output+'/'+label_output0+'/kappa_{0}_{1}.fits'.format(s_,config['nside_out'])
            if not os.path.exists(path_):

                fits_f = Table()
                fits_f['T'] = kappa_lensing[s_]
                fits_f.write(path_)
        print ('save kappa done')

        # make g1 and g2 & IA ---
        for i in frogress.bar(range(kappa_lensing.shape[0])):
            path_ = output+'/'+label_output0+'/gg_{0}_{1}.fits'.format(i,config['nside_out'])

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

    # integrate planes given the redshift distributiins ************************************
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


    for i in frogress.bar(range(2,len(comoving_edges)-1)):

	path_ = output+'/'+label_output0+'/lens_{0}_{1}.fits'.format(i,config['nside_out'])

	pathgg_ = output+'/'+label_output0+'/gg_{0}_{1}.fits'.format(i,config['nside_out'])

	k_ = pf.open(pathgg_)
	d_ = pf.open(path_)
	IA_f = iaa.F_nla(z_centre[i], cosmology.Om0, rho_c1=rho_c1,A_ia = config['A_IA'], eta=config['eta_IA'], z0=config['z0_IA'],  lbar=0., l0=1e-9, beta=0.)
	#print ((k_[1].data['T']))
	for tomo_bin in config['sources_bins']:         
	    m_ = 1.+config['m_sources'][tomo_bin-1]
	    if SC:
		g1_tomo[tomo_bin]  +=  m_*((1.+BIAS_SC*(d_[1].data['T']))*(k_[1].data['g1']+k_[1].data['g1_IA']*IA_f))*nz_kernel_sample_dict[tomo_bin][i]
		g2_tomo[tomo_bin]  +=  m_*((1.+BIAS_SC*(d_[1].data['T']))*(k_[1].data['g2']+k_[1].data['g2_IA']*IA_f))*nz_kernel_sample_dict[tomo_bin][i]
		d_tomo[tomo_bin] +=  (1.+BIAS_SC*d_[1].data['T'])*nz_kernel_sample_dict[tomo_bin][i]
	    else:
		g1_tomo[tomo_bin]  +=  m_*(k_[1].data['g1']+k_[1].data['g1_IA']*IA_f)*nz_kernel_sample_dict[tomo_bin][i]
		g2_tomo[tomo_bin]  +=  m_*(k_[1].data['g2']+k_[1].data['g2_IA']*IA_f)*nz_kernel_sample_dict[tomo_bin][i]
		d_tomo[tomo_bin] +=  (1.+BIAS_SC*d_[1].data['T'])*nz_kernel_sample_dict[tomo_bin][i]





    # need to rotate the maps first -----------------------------------------------------------         
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



    mcal_catalog = load_obj('/global/cfs/cdirs/des/mass_maps/Maps_final/data_catalogs_weighted')

    depth_weigth = np.load('/global/cfs/cdirs/des/mass_maps/Maps_final/depth_maps_Y3_1024_numbdensity.npy',allow_pickle=True).item()


    maps_PKDGRAV = dict()

    sources_maps = dict()
    
    
    
    

    
    for tomo_bin in config['sources_bins']:        



        maps_PKDGRAV[tomo_bin] = dict()
        
        # get ra, dec, w from the des y3 catalog +++++++++++
        dec1 = mcal_catalog[tomo_bin]['dec']
        ra1 = mcal_catalog[tomo_bin]['ra']
        w_ = mcal_catalog[tomo_bin]['w'] 
        

        
        #####################################
        pix_ = convert_to_pix_coord(ra1,dec1, nside=config['nside'])
        mask = np.in1d(np.arange(hp.nside2npix(config['nside'])),pix_)
	
        # in what follows we have different ways to sample the maps.
        if MODE_SAMPLING == 'desy3':
	    # just sample des y3 shear catalog and randomize positions
            pix = pix_
            w = w_
            e1,e2 = mcal_catalog[tomo_bin]['e1'],mcal_catalog[tomo_bin]['e2']
        
        
        elif MODE_SAMPLING == 'randpos_wpos_ell':
                        
            df2 = pd.DataFrame(data = {'w':w_,'pix_':pix_},index = pix_)
            # poisson sample the weight map ------
            n_mean = np.ones(hp.nside2npix(config['nside']))*len(pix_)/len(np.unique(pix_))
            n_mean[~np.in1d(np.arange(len(n_mean)),np.unique(pix_))] = 0
            nn = np.random.poisson(n_mean)
            nn[~mask]= 0



            count = 0

            nnmaxx = max(nn)
            for count in range(nnmaxx):

                pix_valid = np.arange(len(nn))[nn>0]
                nn_valid = nn[nn>0]

                df3 = ((df2.sample(frac=1).drop_duplicates('pix_').sort_index()))#
                df3 = df3.loc[np.unique(pix_valid)]
                if count == 0:
                    w = df3['w']
                    pix = df3['pix_']
                else:
                    w = np.hstack([w,df3['w']])
                    pix = np.hstack([pix,df3['pix_']]) 
                nn -= 1
                
                
                
            # obtain new ellipticities depending on the weight.
            e1,e2 = random_draw_ell_from_w(w,w_,mcal_catalog[tomo_bin]['e1'],mcal_catalog[tomo_bin]['e2'])
        
        elif MODE_SAMPLING == 'randpos_wpos_ell_full':
	
            uu = np.arange(hp.nside2npix(config['nside']))
            n_tot = np.int(hp.nside2npix(config['nside'])*len(pix_)/len(np.unique(pix_)))
            pix = np.random.choice(uu,n_tot)
            w = np.random.choice(w_,len(pix))
                
            # obtain new ellipticities depending on the weight.
            e1,e2 = random_draw_ell_from_w(w,w_,mcal_catalog[tomo_bin]['e1'],mcal_catalog[tomo_bin]['e2'])
        
        
        
        
        elif MODE_SAMPLING == 'randposdepth_wpos_ell': 
	    # this is the most accurate way and the one described in the source clustering draft. it requires a
	    # n_density - depth relation, given by the depth_weight dictionary.
	
            # generate a dataframe --------------
            df2 = pd.DataFrame(data = {'w':w_,'pix_':pix_},index = pix_)
            
            # poisson sample the weight map ------
            nn = np.random.poisson(depth_weigth[tomo_bin])
            nn[~mask]= 0



            count = 0

            nnmaxx = max(nn)
            for count in range(nnmaxx):

                pix_valid = np.arange(len(nn))[nn>0]
                nn_valid = nn[nn>0]

                df3 = ((df2.sample(frac=1).drop_duplicates('pix_').sort_index()))#
                df3 = df3.loc[np.unique(pix_valid)]
                if count == 0:
                    w = df3['w']
                    pix = df3['pix_']
                else:
                    w = np.hstack([w,df3['w']])
                    pix = np.hstack([pix,df3['pix_']]) 
                nn -= 1
            # obtain new ellipticities depending on the weight.
            e1,e2 = random_draw_ell_from_w(w,w_,mcal_catalog[tomo_bin]['e1'],mcal_catalog[tomo_bin]['e2'])
        
 
        # now that we decided how to sample the field, let's make new maps! *****************

        if SC:
            f = 1./np.sqrt(d_tomo[tomo_bin]/np.sum(nz_kernel_sample_dict[tomo_bin]))   #* (1./np.sqrt(depth_weigth[tomo_bin]))
            f = f[pix]
        else:
            f = 1.
            
        n_map = np.zeros(hp.nside2npix(config['nside']))
        n_map_sc = np.zeros(hp.nside2npix(config['nside']))

        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)

        n_map[unique_pix] += np.bincount(idx_rep, weights=w)
        n_map_sc[unique_pix] += np.bincount(idx_rep, weights=w/f**2)

        g1_ = g1_tomo[tomo_bin][pix]
        g2_ = g2_tomo[tomo_bin][pix]


        

        es1,es2 = apply_random_rotation(e1/f, e2/f)
        es1a,es2a = apply_random_rotation(e1/f, e2/f)
        
        
        del mcal_catalog[tomo_bin]
        gc.collect()


        x1_sc,x2_sc = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1,'e2':es2},es_colnames=("e1","e2"))

        e1_map = np.zeros(hp.nside2npix(config['nside']))
        e2_map = np.zeros(hp.nside2npix(config['nside']))
        #
        e1r_map = np.zeros(hp.nside2npix(config['nside']))
        e2r_map = np.zeros(hp.nside2npix(config['nside']))
        
        e1r_map0 = np.zeros(hp.nside2npix(config['nside']))
        e2r_map0 = np.zeros(hp.nside2npix(config['nside']))
        
        g1_map = np.zeros(hp.nside2npix(config['nside']))
        g2_map = np.zeros(hp.nside2npix(config['nside']))
        
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)

        
        
        
        e1_map[unique_pix] += np.bincount(idx_rep, weights= x1_sc*w)
        e2_map[unique_pix] += np.bincount(idx_rep, weights= x2_sc*w)
        
        e1r_map[unique_pix] += np.bincount(idx_rep, weights=es1*w)
        e2r_map[unique_pix] += np.bincount(idx_rep, weights=es2*w)
      
        e1r_map0[unique_pix] += np.bincount(idx_rep, weights=es1a*w)
        e2r_map0[unique_pix] += np.bincount(idx_rep, weights=es2a*w)
        
        
        g1_map[unique_pix] += np.bincount(idx_rep, weights= g1_*w)
        g2_map[unique_pix] += np.bincount(idx_rep, weights= g2_*w)
        
        
        
        
        
        
        
        mask_sims = n_map_sc != 0.
        e1_map[mask_sims]  = e1_map[mask_sims]/(n_map_sc[mask_sims])
        e2_map[mask_sims] =  e2_map[mask_sims]/(n_map_sc[mask_sims])
        e1r_map[mask_sims]  = e1r_map[mask_sims]/(n_map_sc[mask_sims])
        e2r_map[mask_sims] =  e2r_map[mask_sims]/(n_map_sc[mask_sims])
        e1r_map0[mask_sims]  = e1r_map0[mask_sims]/(n_map_sc[mask_sims])
        e2r_map0[mask_sims] =  e2r_map0[mask_sims]/(n_map_sc[mask_sims])
        g1_map[mask_sims]  = g1_map[mask_sims]/(n_map_sc[mask_sims])
        g2_map[mask_sims] =  g2_map[mask_sims]/(n_map_sc[mask_sims])


        EE,BB,_   =  g2k_sphere(g1_map, g2_map, mask_sims, nside=nside, lmax=nside*2 ,nosh=True)
       ## EE,BB,_   =  g2k_sphere(e1r_map0, e2r_map0, mask_sims, nside=nside, lmax=nside*2 ,nosh=True)
        
        #EE,BB,_   =  g2k_sphere(e1_map, e2_map, mask_sims, nside=nside, lmax=nside*2 ,nosh=True)
        
        #''' 
        
	# this apply a stric 2 lma cut on the maps...
        g1_map =  hp.alm2map(hp.map2alm(g1_map, lmax=nside*2) ,nside=nside,  lmax=nside*2)
        g2_map =  hp.alm2map(hp.map2alm(g2_map, lmax=nside*2) ,nside=nside,  lmax=nside*2)
        e1r_map =  hp.alm2map(hp.map2alm(e1r_map, lmax=nside*2) ,nside=nside,  lmax=nside*2)
        e2r_map =  hp.alm2map(hp.map2alm(e2r_map, lmax=nside*2) ,nside=nside,  lmax=nside*2)
        e1r_map0 =  hp.alm2map(hp.map2alm(e1r_map0, lmax=nside*2) ,nside=nside,  lmax=nside*2)
        e2r_map0 =  hp.alm2map(hp.map2alm(e2r_map0, lmax=nside*2) ,nside=nside,  lmax=nside*2)

       # ''' 
    
        EE,BB,_   =  g2k_sphere(g1_map+e1r_map0, g2_map+e2r_map0, mask_sims, nside=nside, lmax=nside*2 ,nosh=True)
        #EE0,BB0,_   =  g2k_sphere(e1r_map0, e2r_map0, mask_sims, nside=nside, lmax=nside*2 ,nosh=True)
        
        #EE += EE0
        
        EEn,BBn,_ =  g2k_sphere(e1r_map, e2r_map, mask_sims, nside=nside, lmax=nside*2 ,nosh=True)

        EE[~mask_sims] = 0.
        EEn[~mask_sims] = 0.

        sources_maps[tomo_bin] = {'EE':EE-np.mean(EE),'EEn':EEn-np.mean(EEn),'mask':mask_sims} 

    save_obj('/global/cfs/cdirs/des/mgatti/temp_maps/{0}'.format(SC_extralab_+nr),sources_maps)



    # this other part computes the moments ++++++++++++++++++++++++++++++++++++++++++
    conf = dict()
    conf['smoothing_scales'] = np.array([3.2,5.1,8.2,13.1,21.0,33.6,54.,86.,138,221.]) # arcmin
    conf['nside'] = 1024
    conf['lmax'] = 2048
    conf['verbose'] = True
    
    output_folder = '/global/cfs/cdirs/des/mgatti//temp_moments/'
    conf['output_folder'] = output_folder + '/output'+SC_extralab_+nr+'/'

    if not os.path.exists(conf['output_folder']):
        try:
            os.mkdir(conf['output_folder'])
        except:
            pass


    #m = load_obj(file.strip('.pkl'))
    mcal_moments = moments_map(conf)
#            
    # assign maps *************
    for i in tomo_bins:
        mcal_moments.add_map(sources_maps[i]['EE'], field_label = 'kE', tomo_bin = i)
        mcal_moments.add_map(sources_maps[i]['EEn'], field_label = 'kN', tomo_bin = i)

    mcal_moments.mask = sources_maps[i]['mask']
    del m 
    gc.collect()

    mcal_moments.transform_and_smooth('convergence','kE',None, shear = False, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)         
    mcal_moments.transform_and_smooth('noise','kN',None, shear = False, tomo_bins = tomo_bins, overwrite = False , skip_loading_smoothed_maps = True)         



    mcal_moments.compute_moments( label_moments='kEkE', field_label1 ='convergence_kE',  tomo_bins1 = tomo_bins)
    #mcal_moments.compute_moments( label_moments='kBkB', field_label1 ='convergence_kB',  tomo_bins1 = tomo_bins)
    mcal_moments.compute_moments( label_moments='kEkN', field_label1 ='convergence_kE', field_label2 = 'noise_kE', tomo_bins1 = tomo_bins)
    mcal_moments.compute_moments( label_moments='kNkN', field_label1 ='noise_kE', field_label2 = 'noise_kE', tomo_bins1 = tomo_bins)
    #mcal_moments.compute_moments( label_moments='kNBkNB', field_label1 ='noise_kB', field_label2 = 'noise_kB', tomo_bins1 = tomo_bins)
    mcal_moments.compute_moments( label_moments='kNkE', field_label2 ='convergence_kE', field_label1 = 'noise_kE',  tomo_bins1 = tomo_bins)

    del mcal_moments.smoothed_maps
    del mcal_moments.fields
    gc.collect()
    save_obj('/global/cfs/cdirs/des/mgatti//temp_moments/output_SC/'+SC_extralab_+nr,mcal_moments)
    os.system('rm -r {0}'.format( conf['output_folder']))
    print ('saving')


                
                
         
            
            
            

    
    
label_output = 'NEW_grid_fiducialcosmo_SC_k_'

# NSIDE OF THE MAPS YOU'LL BE CREATING **********************
nside = 1024
tomo_bins = [0,1,2,3]
SC = True
BIAS_SC = 1.
nr_ = ['_nr1']#,'_nr2','_nr3','_nr4','_nr5','_nr6','_nr7','_nr8','_nr9','_nr10']
# CHANGE THIS TO A FOLDER IN YOUR CSCRATCH1 *****************
output = '/global/cfs/cdirs/des/mgatti/DarkGrid/'

# path to sims
path_sims = '/global/cfs/cdirs/des/darkgrid/fidu_run_1/'
#path_sims = '/global/cfs/cdirs/des/darkgrid/fidu_delta_run_5/'
#path_sims = '/global/cfs/cdirs/des/darkgrid/grid_run_1/'
#path_sims = '/global/cscratch1/sd/dominikz/DES_Y3_PKDGRAV_SIMS/grid_run_1/'


#MODE_SAMPLING = 'desy3'
#SC_extralab_ = 'SC_random_POS1_NOdepth_weight_noremask_'
#
#
#MODE_SAMPLING = 'desy3'
#SC_extralab_ = 'noSC_random_DES_nomask_'
#

#MODE_SAMPLING = 'randposdepth_wpos_ell'
#SC_extralab_ = 'SC_random_depth_wpos_ell_nomask_'


MODE_SAMPLING = 'randposdepth_wpos_ell'
SC_extralab_ = 'SC_random_depth_wpos_ell_nomask_GGN_'
SC_extralab_ = 'SC_random_depth_wpos_ell_withmask_last_'
#srun --nodes=4 --tasks-per-node=20 --cpus-per-task=3 --cpu-bind=cores  python new_convert_PKDGRAV_random.py

#

'''
randpos_wpos_ell_full: full sky random position+depth, weights from the pixel, ellipticities from the P(e|w) distribution
randposdepth_wpos_ell: random position+depth, weights from the pixel, ellipticities from the P(e|w) distribution
randpos_wpos_ell: random position, weights from the pixel, ellipticities from the P(e|w) distribution
desy3: des y3 position, weights and ellipticities
'''
    
        
from mpi4py import MPI 
if __name__ == '__main__':
    
    m = np.load('./params_IA.npy',allow_pickle=True).item()

    import glob
    # /global/cfs/cdirs/des/darkgrid/fidu_run_1/cosmo_Om=0.26_num=0_s8=0.84
    f_ = glob.glob(path_sims+'/cosmo_*')
    
    
    #f_ = ['/global/cfs/cdirs/des/darkgrid/fidu_run_1/cosmo_Om=0.26_num=0_s8=0.84']
    runstodo=[]

    for f in f_:
        
       # if ('cosmo_Om=0.1544' in f) and ('s8=1.2813' in f):
        if ('Om=0.26' in f) and ('s8=0.84' in f):
            
            Omegam = f.split('Om=')[1].split('_')[0]
            s8 = f.split('s8=')[1].split('_')[0]
            num = f.split('num=')[1].split('_')[0]



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
            if extralab:
                ex_ = 'H0='+str(h)+'ns='+str(ns)+'Ob='+str(Ob)
            else:
                ex_=''
            ex_ = 'RANDOM_'+ex_




            p = '{0}_{1}'.format(Omegam,s8)
            A_IA = m['A'][m['om_s8'] == p]
            e_IA = m['E'][m['om_s8'] == p]

            if len(A_IA)==0:
                A_IA = 0.
                e_IA = 0.
            else:
                A_IA = A_IA[0]
                e_IA = e_IA[0]

            label_output1 = label_output+ex_+'Om='+str(Omegam)+'_s8='+str(s8)+'num='+str(num)+'A_IA={0:2.2f}'.format(A_IA)+'e_IA={0:2.2f}'.format(e_IA)
           

            for i in range(4):
                for nr in nr_:
                    
                    p = '//global/cfs/cdirs/des/mgatti//temp_moments/output_SC//'+SC_extralab_+label_output1+'_'+str(i+1)+nr

                    #/pscratch/sd/m/mgatti/temp_moments/output_SC/'+SC_extralab_+nr,mcal_moments)
                    if not os.path.exists(p+'.pkl'):
                        runstodo.append([f,i,label_output1+'_'+str(i+1)+nr])








    print ('RUNSTODO: ',len(runstodo))
    #make_maps(runstodo[run_count])
    run_count=0
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
##
##





