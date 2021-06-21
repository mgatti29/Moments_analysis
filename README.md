# Moments_analysis

This repo contains the code to replicate the results in https://arxiv.org/pdf/1911.05568.pdf
I am working on the documentation but it might take a while.


# Installation.

You first need to run python setup.py install in the repo folder.
The other packages you might need to install are:

- emcee
- h5py
- pyfits
- healpy
- pandas
- george
- xarray


# cosmosis setup to run chains.

The current chains can only run at nersc - there's a pre-compiled script in the code folder that'll work only on nersc (unless you compile it, but that's not stable enough yet).

In your cosmosis folders, please do:
- cosmosis:
git checkout develop
- cosmosis-standard-library:
git checkout des-y3
- cosmosis-des-library:
git checkout develop
