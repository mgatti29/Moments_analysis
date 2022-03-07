import numpy as np
import healpy as hp


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
