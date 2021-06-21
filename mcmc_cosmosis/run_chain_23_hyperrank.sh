#!/bin/bash

#SBATCH --qos=regular
#SBATCH --nodes=8
#SBATCH --license=cscratch1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=256
#SBATCH --constraint=haswell
#SBATCH --time=48:00:00

export RUNNAME='2_3tomocross_kEkE_flask_24_1000_reg'
export KEY_moments='2_3tomocross_kEkE_flask_24_1000_reg'
export DATAFILE=/project/projectdirs/des/www/y3_chains/data_vectors/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_6000HR.fits
export DATAFILE_SR=data/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_sr.npy
export SCALEFILE=Y3_params_priors/scales.ini
export INCLUDEFILE=Y3_params_priors/params_t17_hyp.ini
export VALUESINCLUDE=Y3_params_priors/values_SR_hyp.ini
export PRIORSINCLUDE=Y3_params_priors/priors.ini

# the following lines you need to change them such that it loads your environment & cosmosis and then move to the mcmc_cosmosis folder. Note that you also need to change the following path to your installation of Moments_analysis.
export moments_like_path=/global/u2/m/mgatti/Moments_analysis/mcmc_cosmosis/moment_likelihood.py

cd //global/homes/m/mcraveri/
source cosmosis/config/setup-cosmosis-nersc 3
cd /global/homes/m/mcraveri/Moments_analysis/mcmc_cosmosis


srun cosmosis --mpi  Y3_params_priors/params_hyp.ini  -p runtime.sampler='polychord' 

# you can also just test it works by opening the terminal, exporting the variable above and running the following line
# cosmosis Y3_params_priors/params.ini 









