import gc
import pyfits as pf
import pickle
import numpy as np
from mpi4py import MPI 
import healpy as hp
import os
import copy
from Moments_analysis import convert_to_pix_coord, IndexToDeclRa, apply_random_rotation, addSourceEllipticity, gk_inv
import healpy as hp
import scipy
from scipy.interpolate import interp1d
import timeit
import glob
import UFalcon
import ekit
from ekit import paths as path_tools
import frogress
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from UFalcon import utils
from UFalcon import constants as constants_u



'''
This code reads the output of PKDGRAV simulations (which is made of matter particles as a function of ra,dec,z),
and creates convergence maps assuming the DES Y3 n(z).
It then convert the convergence maps to shear maps and add DES Y3 noise.


WHERE THE SIMULATIONS ARE.

We have 50 full sky simulations at fixed cosmology here:
/global/cscratch1/sd/dominikz/DES_Y3_PKDGRAV_SIMS/fidu_run_1/cosmo_Om=0.26_num=*_s8=0.84

from each of those you can get 4 independent DES Y3 footprint; this means have 200 independent realisations to compute our covariance matrix.

We then have simulations at different cosmological parameter here:
/global/cscratch1/sd/dominikz/DES_Y3_PKDGRAV_SIMS/grid_run_1> cosmo_Om=*_num=*_s8=*

for each different combination of Om and s8, you have 5 full sky simulations. This means you'll have ~20 independent des y3 realisations.


If I remember well, you can't run more than 2-3 jobs per node; so you'll be using something like
srun --nodes=1 --tasks-per-node=3 --cpus-per-task=20 --cpu-bind=cores --mem=110GB python convert_PKDGRAV_2maps.py

'''


# you probably want to use this label to specify the cosmological parameter of the sims
label_output = 'fid'

# this has to be 5 if you're using the sims at different cosmological values in the /grid_run_1/ folder, or to 50 for the sims in /fidu_run_1/
chunk = 5


# NSIDE OF THE MAPS YOU'LL BE CREATING **********************
nside = 1024

# NUMBER OF SIMULATIONS *************************************
nsims = 20

# CHANGE THIS TO A FOLDER IN YOUR CSCRATCH1 *****************
output = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/PKDGRAV_tests/'


# path to sims
path_sims = '/global/cscratch1/sd/dominikz/DES_Y3_PKDGRAV_SIMS/fidu_run_1/'


############## COSMOLOGICAL PARAMETERS ################################################################
# change them depending on the sim you're using!
Omegam = 0.26
s8=0.84


h = 0.673 #don't change this









class Continuous:
    """
    Computes the lensing weights for a continuous, user-defined n(z) distribution.
    """

    def __init__(self, n_of_z, z_lim_low=0, z_lim_up=None, shift_nz=0.0, IA=0.0):
        """
        Constructor.
        :param n_of_z: either path to file containing n(z), assumed to be a text file readable with numpy.genfromtext
                        with the first column containing z and the second column containing n(z), or a callable that
                        is directly a redshift distribution
        :param z_lim_low: lower integration limit to use for n(z) normalization, default: 0
        :param z_lim_up: upper integration limit to use for n(z) normalization, default: last z-coordinate in n(z) file
        :param shift_nz: Can shift the n(z) function by some redshift (intended for easier implementation of photo z bias)
        :param IA: Intrinsic Alignment. If not None computes the lensing weights for IA component
                        (needs to be added to the weights without IA afterwards)
        """

        # we handle the redshift dist depending on its type
        if callable(n_of_z):
            if z_lim_up is None:
                raise ValueError("An upper bound of the redshift normalization has to be defined if n_of_z is not a "
                                 "tabulated function.")

            self.nz_intpt = n_of_z
            # set the integration limit and integration points
            self.lightcone_points = None
            self.limit = 1000
        else:
            # read from file
            nz = np.genfromtxt(n_of_z)

            # get the upper bound if necessary
            if z_lim_up is None:
                z_lim_up = nz[-1, 0]

            # get the callable function
            self.nz_intpt = interp1d(nz[:, 0] - shift_nz, nz[:, 1], bounds_error=False, fill_value=0.0)

            # points for integration
            self.lightcone_points = nz[np.logical_and(z_lim_low < nz[:, 0], nz[:, 0] < z_lim_up), 0]

            # check if there are any points remaining for the integration
            if len(self.lightcone_points) == 0:
                self.lightcone_points = None
                self.limit = 1000
            else:
                self.limit = 10 * len(self.lightcone_points)

        self.z_lim_up = z_lim_up
        self.z_lim_low = z_lim_low
        self.IA = IA
        # Normalization
        self.nz_norm = integrate.quad(lambda x: self.nz_intpt(x), z_lim_low, self.z_lim_up,
                                      points=self.lightcone_points, limit=self.limit)[0]

    def __call__(self, z_low, z_up, cosmo, lens = False):
        """
        Computes the lensing weights for the redshift interval [z_low, z_up].
        :param z_low: lower end of the redshift interval
        :param z_up: upper end of the redshift interval
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: lensing weight
        """
        if lens:
            norm = (z_up - z_low) * self.nz_norm
            # lensing weights
            def f(x):
                return (self.nz_intpt(x))

            if self.lightcone_points is not None:
                numerator = integrate.quad(f, z_low, z_up, points=self.lightcone_points[np.logical_and(z_low < self.lightcone_points, self.lightcone_points < z_up)])[0]
            else:
                numerator = integrate.quad(f, z_low, z_up)[0]
            return numerator / norm
        else:
            norm = utils.dimensionless_comoving_distance(z_low, z_up, cosmo) * self.nz_norm
            norm *= (utils.dimensionless_comoving_distance(0., (z_low + z_up)/2., cosmo) ** 2.)
            if abs(self.IA - 0.0) < 1e-10:
                # lensing weights without IA
                numerator = integrate.quad(self._integrand_1d, z_low, z_up, args=(cosmo,))[0]
            else:
                # lensing weights for IA
                numerator = (2.0/(3.0*cosmo.Om0)) * \
                            w_IA(self.IA, z_low, z_up, cosmo, self.nz_intpt, points=self.lightcone_points)

            return numerator / norm

    def _integrand_2d(self, y, x, cosmo):
        """
        The 2d integrant of the continous lensing weights
        :param y: redhsift that goes into the n(z)
        :param x: redshift for the Dirac part
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: the 2d integrand function
        """
        return self.nz_intpt(y) * \
               utils.dimensionless_comoving_distance(0, x, cosmo) * \
               utils.dimensionless_comoving_distance(x, y, cosmo) * \
               (1 + x) * \
               cosmo.inv_efunc(x) / \
               utils.dimensionless_comoving_distance(0, y, cosmo)

    def _integrand_1d(self, x, cosmo):
        """
        Function that integrates out y from the 2d integrand
        :param x: at which x (redshfit to eval)
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: the 1d integrant at x
        """
        if self.lightcone_points is not None:
            points = self.lightcone_points[np.logical_and(self.z_lim_low < self.lightcone_points,
                                                          self.lightcone_points < self.z_lim_up)]
            quad_y = lambda x: integrate.quad(lambda y: self._integrand_2d(y, x, cosmo), x, self.z_lim_up,
                                              limit=self.limit, points=points)[0]
        else:
            quad_y = lambda x: integrate.quad(lambda y: self._integrand_2d(y, x, cosmo), x, self.z_lim_up,
                                              limit=self.limit)[0]

        return quad_y(x)

def w_IA(IA, z_low, z_up, cosmo, nz_intpt, points=None):
    """
    Calculates the weight per slice for the NIA model given a
    distribution of source redshifts n(z).
    :param IA: Galaxy Intrinsic alignment amplitude
    :param z_low: Lower redshift limit of the shell
    :param z_up: Upper redshift limit of the shell
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :param nz_intpt: nz function
    :param points: Points in redshift where integrand is evaluated (used for better numerical integration), can be None
    :return: Shell weight for NIA model

    """

    def f(x):
        return (F_NIA_model(x, IA, cosmo) * nz_intpt(x))

    if points is not None:
        dbl = integrate.quad(f, z_low, z_up, points=points[np.logical_and(z_low < points, points < z_up)])[0]
    else:
        dbl = integrate.quad(f, z_low, z_up)[0]

    return dbl


def density_prefactor(n_pix, n_particles, boxsize, cosmo):
    """
    Computes the prefactor to transform from number of particles to convergence, see https://arxiv.org/abs/0807.3651,
    eq. (A.1).
    :param n_pix: number of healpix pixels used
    :param n_particles: number of particles
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :return: convergence prefactor
    """
    convergence_factor = (n_pix / (4.0 * np.pi)) * \
                         (cosmo.H0.value / constants_u.c) ** 2 * \
                         (boxsize * 1000.0) ** 3 / n_particles
    return convergence_factor


def get_parameters_from_path(paths, suffix=True, fmt=None):
    """
    Given a list of standardised paths, or a single path created with
    create_path() this function reads the parameters in the paths.

    :param paths: Either a single string or a list of strings. The strings
                  should be paths in the create_path() format.
    :param suffix: If True assumes that the given paths have suffixes and
                   exclues them from the parsing
    :return: Returns a dictionary which contains the defined parameters and
             a list containing the undefined parameters.
    """
    # convert to list if needed
    if not isinstance(paths, list):
        paths = [paths]

    # use first path to initialize the dictionary and list for output
    defined_names = []
    undefined_count = 0
    path = paths[0]

    path = _prepare_path(path, suffix=suffix)

    # loop over parameters in first path to initialize dictionary
    for c in path:
        if isinstance(c, list):
            c = c[0]
        if '=' in c:
            b = c.split('=')
            defined_names.append(b[0])
        else:
            undefined_count += 1

    # initialize
    undefined = np.zeros((len(paths), undefined_count), dtype=object)
    defined = {}
    for d in defined_names:
        defined[d] = np.zeros(len(paths), dtype=object)

    # loop over files and get parameters
    for ii, path in enumerate(paths):
        path = _prepare_path(path, suffix=suffix)
        count = 0
        for idx_c, c in enumerate(path):
            if isinstance(c, list):
                c = c[0]
            if '=' in c:
                b = c.split('=')
                to_add = _check_type(b[1], fmt, idx_c)
                defined[b[0]][ii] = to_add
            else:
                to_add = _check_type(c, fmt, idx_c)
                undefined[ii, count] = to_add
                count += 1
    return defined, undefined




def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f)
        f.close()
    return mute




  
    
    

def rotate_map_approx(mask, rot_angles, flip=False):
    alpha, delta = hp.pix2ang(2048, np.arange(len(mask)))

    rot = hp.rotator.Rotator(rot=rot_angles, deg=True)
    rot_alpha, rot_delta = rot(alpha, delta)
    if not flip:
        rot_i = hp.ang2pix(2048, rot_alpha, rot_delta)
    else:
        rot_i = hp.ang2pix(2048, np.pi-rot_alpha, rot_delta)
    rot_map = mask*0.
    rot_map[rot_i] =  mask[np.arange(len(mask))]
    return rot_map



def lens_weights(n_of_z):
    nz = np.genfromtxt(n_of_z)
    nz_intpt = interp1d(nz[:, 0], nz[:, 1])
    nz_norm = np.trapz(nz[:, 1],nz[:, 0])
    
    
def make_maps(seed):
    print (output+'seed_'+str(seed+1))
    st = timeit.default_timer()
    import healpy as hp
    import pyfits as pf
    density_full_maps = dict()
    density_maps = dict()
    lenses_maps = dict()
    random_maps = dict()
    sources_maps = dict()
    sources_cat = dict()
    
    
    mask_DES_y3 = load_obj('/project/projectdirs/des/mass_maps/Maps_final//mask_DES_y3')


    
    # defines cosmology for PKDGRAV sims
    from astropy.cosmology import FlatLambdaCDM
    constants = FlatLambdaCDM(H0=float(h * 100.), Om0=float(Omegam))
    
    aa=0
    if seed <=chunk:
        rel = seed
    elif (seed>chunk) &  (seed <=chunk*2):
        rel = seed-chunk
        aa=1
    elif (seed>chunk*2) &  (seed <=chunk*3):
        
        rel = seed-chunk*2
        aa=2
    elif (seed>chunk*3) &  (seed <=chunk*4):
        rel = seed-chunk*3
        aa=3#
        
        
        

    print (seed,rel)
    
    shell_directory = path_sims+'/cosmo_Om={1:2.f}_num={0}_s8={2:2.f}'.format(rel,Omegam,s8)
    
    
    maps_base = dict()
    lens_cat = dict()
    random_cat = dict()

        
    # load unblinded Nz ******************************
    for i in range(4):
        kappa = np.zeros((1, hp.nside2npix(2048)), dtype=np.float32)            
        path_nz ='/global/project/projectdirs/des/mgatti/Moments_analysis/Nz/FLASK_{0}.txt'.format(i+1)
        lensing_weights = UFalcon.lensing_weights.Continuous(
                path_nz, z_lim_low=0., z_lim_up=2., shift_nz=0., IA=0.)
        
        
        shell_files = glob.glob("{}DES-Y3-shell*".format(shell_directory))
        z_bounds, _ = path_tools.get_parameters_from_path(shell_files)
        for zz in frogress.bar(range(len(z_bounds['z-low']))):
            shell = hp.read_map(shell_files[zz])

            # conversion from particle counts to convergence
            kappa_prefactor = UFalcon.lensing_weights.kappa_prefactor(
                            n_pix=shell.size, n_particles=768**3,
                            boxsize=1.33610451306, cosmo=constants)
            shell *= kappa_prefactor

            # Add the shell multiplied by the lensing weights
            weight = lensing_weights(
                z_bounds['z-low'][zz], z_bounds['z-high'][zz], constants)
            shell *= weight

            kappa += shell

        # Subtract mean
        mean = np.mean(kappa)
        kappa -= mean

        conv_map_k1=kappa[0]
    
        conv_map_k = copy.copy(conv_map_k1)
        # include some rotation here **********************
        if (seed+aa)%4 ==0:
            pass
        elif (seed+aa)%4 ==1:
            print (1)
            conv_map_k = rotate_map_approx(conv_map_k,[ 180 ,0 , 0], flip=False)
        elif (seed+aa)%4 ==2:
            print (2)
            conv_map_k = rotate_map_approx(conv_map_k,[ 90 ,0 , 0], flip=True)
        elif (seed+aa)%4 ==3:
            print (3)
            conv_map_k = rotate_map_approx(conv_map_k,[ 270 ,0 , 0], flip=True)
        
        

        
        # *******************************
        maps_base[i] = conv_map_k
        
        
        
    print (maps_base.keys())
        
        
        
        
        


    
    
    
    del conv_map_k
    gc.collect()
    mcal_catalog = load_obj('/project/projectdirs/des/mass_maps/Maps_final/data_catalogs_weighted')
   
    for i in range(4):
        # load_FLASK
 
        g1,g2 = gk_inv(maps_base[i],maps_base[i]*0.,2048,2048*2)
        
        #g1_1024_full = hp.alm2map(hp.map2alm(g1,lmax = 2048),nside = 1024)
        #g2_1024_full = hp.alm2map(hp.map2alm(g2,lmax = 2048),nside = 1024)
        #
        #g1_1024 = copy.copy(g1_1024_full)
        #g2_1024 = copy.copy(g2_1024_full)
        #g1_1024[~mask_DES_y3] = 0
        #g2_1024[~mask_DES_y3] = 0
        #
        
        gc.collect()
        
        dec1 = mcal_catalog[i]['dec']
        ra1 = mcal_catalog[i]['ra']
        w = mcal_catalog[i]['w']
        print (len(w))
        
        es1,es2 = apply_random_rotation(mcal_catalog[i]['e1'], mcal_catalog[i]['e2'])
        pix = convert_to_pix_coord(ra1,dec1, nside=2048)
        g1 = g1[pix]
        g2 = g2[pix]
        kk = maps_base[i][pix]
        del maps_base[i]
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
    
        sources_cat[i] = {'ra': ra1, 'dec': dec1,'w':w, 'e1':x1,'e2':x2,'g1':g1,'g2':g2}
        sources_maps[i] = {'k_orig_map':k_orig,'g1_field_full':1,'g2_field_full':1,'g1_field':1,'g2_field':1,'g1_map':g1_map,'g2_map':g2_map,'e1_map':e1_map,'e2_map':e2_map,'e1r_map':e1_map_rndm,'e2r_map':e2_map_rndm,'N_source_map':nn,'N_source_mapw':n_map}
        
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
    end = timeit.default_timer()
    print (end-st)

    save_obj(output+'seed_'+'_'+label_output+'_'+str(seed+1),{'sources':sources_maps}) ##,'lens':maps_base_lens,'randoms':random_maps})

        
if __name__ == '__main__':
    runstodo=[]
    for i in range(nsims):
        if not os.path.exists(output+'seed_'+'_'+label_output+'_'+str(i+1)+'.pkl'):
            runstodo.append(i)
    run_count=0
    print (runstodo)
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
        
        
        
       
