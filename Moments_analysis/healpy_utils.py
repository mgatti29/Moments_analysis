import numpy as np
import healpy as hp

def apply_random_rotation(e1_in, e2_in):
    """
    Applies a random rotation to the input ellipticities.

    Args:
        e1_in (array): Input ellipticities (component 1).
        e2_in (array): Input ellipticities (component 2).

    Returns:
        tuple: Rotated ellipticities.
    """
    np.random.seed()
    rot_angle = np.random.rand(len(e1_in)) * 2 * np.pi
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = e1_in * cos + e2_in * sin
    e2_out = -e1_in * sin + e2_in * cos
    return e1_out, e2_out

def IndexToDeclRa(index, nside, nest=False):
    """
    Converts HEALPix index to Declination and Right Ascension.

    Args:
        index (array): HEALPix pixel indices.
        nside (int): HEALPix nside parameter.
        nest (bool, optional): Nesting scheme of the HEALPix pixels. Defaults to False.

    Returns:
        tuple: Declination and Right Ascension.
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)
    return -np.degrees(theta - np.pi / 2.), np.degrees(phi)

def convert_to_pix_coord(ra, dec, nside=1024):
    """
    Converts RA, DEC to HEALPix pixel coordinates.

    Args:
        ra (array): Right Ascension values.
        dec (array): Declination values.
        nside (int, optional): HEALPix nside parameter. Defaults to 1024.

    Returns:
        array: HEALPix pixel coordinates.
    """
    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    return pix

def generate_randoms_radec(minra, maxra, mindec, maxdec, Ngen, raoffset=0):
    """
    Generates random Right Ascension and Declination values within specified ranges.

    Args:
        minra (float): Minimum Right Ascension value.
        maxra (float): Maximum Right Ascension value.
        mindec (float): Minimum Declination value.
        maxdec (float): Maximum Declination value.
        Ngen (int): Number of random points to generate.
        raoffset (float, optional): Right Ascension offset. Defaults to 0.

    Returns:
        tuple: Randomly generated Right Ascension and Declination values.
    """
    r = 1.0
    zmin = r * np.sin(np.pi * mindec / 180.)
    zmax = r * np.sin(np.pi * maxdec / 180.)
    phimin = np.pi / 180. * (minra - 180 + raoffset)
    phimax = np.pi / 180. * (maxra - 180 + raoffset)
    z_coord = np.random.uniform(zmin, zmax, Ngen)
    phi = np.random.uniform(phimin, phimax, Ngen)
    dec_rad = np.arcsin(z_coord / r)
    ra = phi * 180 / np.pi + 180 - raoffset
    dec = dec_rad * 180 / np.pi
    return ra, dec

def addSourceEllipticity(es, es_colnames=("e1","e2"), rs_correction=True, inplace=False):
    """
    Adds intrinsic source ellipticity to shear measurements.

    Args:
        es (array): Array of intrinsic ellipticities.
        es_colnames (tuple, optional): Column names for the intrinsic ellipticities. Defaults to ("e1", "e2").
        rs_correction (bool, optional): Flag indicating whether to apply Rousseeuw and Schneider correction. Defaults to True.
        inplace (bool, optional): Flag indicating whether to modify the input shear measurements in-place. Defaults to False.

    Returns:
        tuple: Modified shear measurements.
    """
    assert len(self) == len(es)
    es_c = np.array(es[es_colnames[0]] + es[es_colnames[1]] * 1j)
    g = np.array(self["shear1"] + self["shear2"] * 1j)
    e = es_c + g
    if rs_correction:
        e /= (1 + g.conjugate() * es_c)
    if inplace:
        self["shear1"] = e.real
        self["shear2"] = e.imag
    else:
        return (e.real, e.imag)

def gk_inv(K, KB, nside, lmax):
    """
    Performs inverse spin transformation on convergence and curl components.

    Args:
        K (array): Convergence map.
        KB (array): Curl/B-mode map.
        nside (int): HEALPix nside parameter.
        lmax (int): Maximum multipole moment.

    Returns:
        tuple: Reconstructed E-mode and B-mode maps.
    """
    alms = hp.map2alm(K, lmax=lmax, pol=False)
    ell, emm = hp.Alm.getlm(lmax=lmax)
    kalmsE = alms / (1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
    kalmsE[ell == 0] = 0.0
    alms = hp.map2alm(KB, lmax=lmax, pol=False)
    ell, emm = hp.Alm.getlm(lmax=lmax)
    kalmsB = alms / (1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
    kalmsB[ell == 0] = 0.0
    _, e1t, e2t = hp.alm2map([kalmsE, kalmsE, kalmsB], nside=nside, lmax=lmax, pol=True)
    return e1t, e2t

def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048, nosh=True):
    """
    Converts shear to convergence on a sphere using HEALPix maps.

    Args:
        gamma1 (array): Shear component 1 map.
        gamma2 (array): Shear component 2 map.
        mask (array): Mask map.
        nside (int, optional): HEALPix nside parameter. Defaults to 1024.
        lmax (int, optional): Maximum multipole moment. Defaults to 2048.
        nosh (bool, optional): Flag indicating whether to remove the spin-2 component. Defaults to True.

    Returns:
        tuple: Convergence (E-mode) map, Curl (B-mode) map, E-mode alm coefficients.
    """
    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask
    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)
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
