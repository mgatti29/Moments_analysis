#!/bin/bash

#SBATCH --qos=regular
#SBATCH --nodes=6
#SBATCH --license=cscratch1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=128
#SBATCH --constraint=haswell
#SBATCH --time=48:00:00

export RUNNAME='2_3_final_flask_compress_t17_24_SR'
export KEY_moments='2_3_final_flask_compress_t17_24_SR'
export DATAFILE=data/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits
export DATAFILE_SR=data/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_sr.npy
export SCALEFILE=Y3_params_priors/scales.ini
export INCLUDEFILE=Y3_params_priors/params_t17.ini
export VALUESINCLUDE=Y3_params_priors/values_SR.ini
export PRIORSINCLUDE=Y3_params_priors/priors.ini

# the following lines you need to change them such that it loads your environment & cosmosis and then move to the mcmc_cosmosis folder. Note that you also need to change the following path to your installation of Moments_analysis.
export moments_like_path=/global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis/moment_likelihood.py

cd /global/cscratch1/sd/mgatti/Mass_Mapping/
source cosmosis/config/setup-cosmosis-nersc 3
cd /global/u2/m/mgatti/Mass_Mapping/Moments_analysis/mcmc_cosmosis
source activate py3s_2


srun cosmosis --mpi  Y3_params_priors/params.ini  -p runtime.sampler='polychord' 

# you can also just test it works by opening the terminal, exporting the variable above and running the following line
# cosmosis Y3_params_priors/params_t17.ini 