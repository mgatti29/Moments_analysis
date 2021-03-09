import ctypes
from ctypes import *
lib_theo=ctypes.cdll.LoadLibrary('/global/homes/m/mgatti/Mass_Mapping/Bihalofit/theory_code/trough.so')
lib_theo.bispectrum.argtypes = [c_double,  c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double] # output pointer
lib_theo.bispectrum.restype  = c_double

h=0.6727;     #// Hubble parameter
sigma8=0.831; #// sigma 8
omb=0.0492;   #// Omega baryon
omc=0.2664;   #// Omega CDM
ns=0.9645;    #// spectral index of linear P(k)
w=-1.0;       #// equation of state of dark energy
z = 0.4
k1=3.
k2=2.
k3=1.5
# pass parameters




print lib_theo.bispectrum(c_double(omc),c_double(omb),c_double(sigma8),c_double(ns),c_double(h),c_double(z),c_double(k1),c_double(k2),c_double(k3),c_int(0))


#
