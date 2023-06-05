# Moments_analysis

This repository is a comprehensive toolkit for performing an end-to-end cosmological analysis using moments of weak lensing mass maps. It provides functionalities to compute theoretical predictions, measure second and third moments, and perform a complete mcmc analysis of weak lensing mass maps moments.
This repository contains the code to replicate the results in https://arxiv.org/pdf/1911.05568.pdf

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
- cosmolopy==0.4.1
- pys2let


# cosmosis setup to run chains.

The current chains can only run at nersc - there's a pre-compiled script in the code folder that'll work only on nersc (unless you compile it, but that's not stable enough yet).

In your cosmosis folders, please do:
- cosmosis:
git checkout develop
- cosmosis-standard-library:
git checkout des-y3
- cosmosis-des-library:
git checkout develop

# Contributing
Contributions to this project are welcome! If you encounter any issues, have ideas for improvements, or would like to add new features, please submit a pull request or open an issue on the GitHub repository.
