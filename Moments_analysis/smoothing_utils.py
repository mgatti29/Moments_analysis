import scipy
import scipy.special
import healpy as hp
import numpy as np

class Alm(object):
    """This class provides some static methods for alm index computation.
    * getlm(lmax,i=None)
    * getidx(lmax,l,m)
    * getsize(lmax,mmax=-1)
    * getlmax(s,mmax=-1)
    """
    def __init__(self,lmax):
        pass

    @staticmethod
    def getlm(lmax,i=None):
        """Get the l and m from index and lmax.
        
        Parameters:
        - lmax
        - i: the index. If not given, the function return l and m
             for i=0..Alm.getsize(lmax)
        """
        if i is None:
            i=np.arange(Alm.getsize(lmax))
        m=(np.ceil(((2*lmax+1)-np.sqrt((2*lmax+1)**2-8*(i-lmax)))/2)).astype(int)
        l = i-m*(2*lmax+1-m)/2
        return (l,m)

    @staticmethod
    def getidx(lmax,l,m):
        """Get index from lmax, l and m.
        
        Parameters:
        - lmax
        - l
        - m
        """
        return m*(2*lmax+1-m)/2+l

    @staticmethod
    def getsize(lmax,mmax=-1):
        if mmax<0 or mmax > lmax:
            mmax=lmax
        return mmax*(2*lmax+1-mmax)/2+lmax+1

    @staticmethod
    def getlmax(s,mmax=-1):
        if mmax >= 0:
            x=(2*s+mmax**2-mmax-2)/(2*mmax+2)
        else:
            x=(-3+np.sqrt(1+8*s))/2
        if x != np.floor(x):
            return -1
        else:
            return int(x)

def almxfl(alm,fl,mmax=-1,inplace=False):
    """Multiply alm by a function of l. The function is assumed
    to be zero where not defined.
    If inplace is True, the operation is done in place. Always return
    the alm array (either a new array or the modified input array).
    """
    # this is the expected lmax, given mmax
    lmax = Alm.getlmax(alm.size,mmax)
    if lmax < 0:
        raise TypeError('Wrong alm size for the given mmax.')
    if mmax<0:
        mmax=lmax
    fl = np.array(fl)
    if inplace:
        almout = alm
    else:
        almout = alm.copy()
    for l in range(lmax+1):
        if l < fl.size:
            a=fl[l]
        else:
            a=0
        i=(Alm.getidx(lmax,l,np.arange(min(mmax,l)+1))).astype(np.int)
        almout[i] *= a
    return almout

