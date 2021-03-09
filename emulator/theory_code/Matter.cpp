
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sf_legendre.h>

//#include "H5Cpp.h"
#include "Matter.h"
#include "Eisenstein_and_Hu.h"
#include "generating_function_utils.h"
#include "Matter_configuration.h"
#include "Matter_integrands.h"
#include "Matter_halofit.h"
#include "Matter_variances.h"
#include "Matter_output.h"

//using namespace H5;

/* _____ Content Matter.cpp
 * 
 * _____ 1. Initialization
 * 
 * ..... 1.1 Constructor
 * ..... 1.2 Destructor
 * ..... 1.3 set_wave_numbers
 * ..... 1.4 set_transfer_function_Eisenstein_and_Hu
 * ..... 1.5 change_cosmology (1)
 * ..... 1.6 change_cosmology (2)
 * 
 * _____ 3. Growth Factors and Power Spectra
 * 
 * ..... 3.2 initialize_linear_growth_factor_of_delta
 * ..... 3.5 Newtonian_linear_power_spectrum
 * ..... 3.8 variance_of_matter_within_R
 * 
 * _____ 5. Output and Checks
 * 
 * ..... 5.2 print_Newtonian_growth_factor
 * ..... 5.6 return_wave_numbers
 * ..... 5.7 transfer_function_at
 * 
 */


/*******************************************
 *******************************************
 **__________ 1. INITIALIZATION __________**
 *******************************************
 ******************************************* 
 *                                         *
 * ..... 1.1 Constructor                   *
 * ..... 1.2 Destructor                    *
 * ..... 1.3 set_wave_numbers              ****************
 * ..... 1.4 set_transfer_function_Eisenstein_and_Hu      *
 * ..... 1.5 set_transfer_function_Bond_and_Efstathiou    *
 * ..... 1.6 set_cylinder_variances        ****************
 *                                         *
 *******************************************
 *******************************************/

/*******************************************************************************************************************************************************
 * 1.1
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

Matter::Matter(Universe* uni, double s3_enhance){
  
  //this->load_Mead_power();
  //this->load_CAMB_power();
  this->S3_enhance = s3_enhance;
  this->universe = uni;
  this->cosmology = this->universe->return_cosmology();  
  
  this->eta_initial = this->universe->return_eta(0);
  this->eta_final = this->universe->return_eta(this->universe->return_number_of_entries()-1);
  this->order = 4;  
  this->set_wave_numbers();

  double a = this->universe->a_at_eta(this->eta_initial);
  double a_prime = a*this->universe->H_at_eta(this->eta_initial);
  this->initialize_linear_growth_factor_of_delta(a, a_prime);
  
  double D1_initial = this->Newtonian_growth_factor_of_delta[0];
  double D1_prime_initial = this->Newtonian_growth_factor_of_delta_prime[0];
  this->initialize_up_to_second_order_growth_factor_of_delta(2.0/7.0*D1_initial*D1_initial, 4.0/7.0*D1_initial*D1_prime_initial);

  this->set_transfer_function_Eisenstein_and_Hu();

  this->norm = this->variance_of_matter_within_R(8.0);
  this->set_cylinder_variances();

}

/*******************************************************************************************************************************************************
 * 1.2 Destructor
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

Matter::~Matter(){
}


/*******************************************************************************************************************************************************
 * 1.3 set_wave_numbers
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

void Matter::set_wave_numbers(){
  
  log_binning(minimal_wave_number*c_over_e5, maximal_wave_number*c_over_e5, number_of_k-1, &this->wave_numbers);

  int n = this->wave_numbers.size();

  this->log_wave_numbers.resize(n);

  for(int i = 0; i<n; i++){
    this->log_wave_numbers[i] = log(this->wave_numbers[i]);
  }

}

/*******************************************************************************************************************************************************
 * 1.4 set_transfer_function_Eisenstein_and_Hu
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

void Matter::set_transfer_function_Eisenstein_and_Hu(){
  
  tranfer_function_Eisenstein_and_Hu(&this->wave_numbers, &this->transfer_function, this->cosmology.Omega_m, this->cosmology.Omega_b, this->cosmology.h_100, this->cosmology.theta_27);  

}


/*************************************************************
 *************************************************************
 **__________ 3. GROWTH FACTORS AND POWER SPECTRA __________**
 *************************************************************
 ************************************************************* 
 *                                                           *
 * ..... 3.1 initialize_linear_growth_factor_of_delta        *
 * ..... 3.2 Newtonian_linear_power_spectrum                 *
 * ..... 3.3 variance_of_matter_within_R                     *
 * ..... 3.4 derivative_variance_of_matter_within_R          *
 * ..... 3.5 variance_of_matter_within_R_2D                  *
 * ..... 3.6 variance_of_matter_within_R_2D_NL               *
 * ..... more ......                                         *
 *                                                           *
 *************************************************************
 *************************************************************/


/*******************************************************************************************************************************************************
 * 3.1 initialize_linear_growth_factor_of_delta
 * Description:
 *
 * Arguments:
 * 
 * Comments:
 * - In this method there are still NO radiation inhomogeneities includes. This would also require to take into account
 *   entropy perturbation!
 * 
*******************************************************************************************************************************************************/

void Matter::initialize_linear_growth_factor_of_delta(double D, double D_prime){

  integration_parameters params;
  params.pointer_to_Matter = this;
  params.pointer_to_Universe = this->universe;
  params.Omega_m = this->cosmology.Omega_m;
  integration_parameters * pointer_to_params = &params;
  
  gsl_odeiv2_system sys = {growth_factor_gsl, growth_factor_gsl_jac, 2, (void *) pointer_to_params};

  this->number_of_entries_Newton = len_ww;
  
  this->eta_Newton.resize(this->number_of_entries_Newton);
  this->Newtonian_growth_factor_of_delta.resize(this->number_of_entries_Newton);
  this->Newtonian_growth_factor_of_delta_prime.resize(this->number_of_entries_Newton);
  this->Newtonian_growth_factor_of_delta_prime_prime.resize(this->number_of_entries_Newton);
  this->Newtonian_growth_factor_of_delta_prime_prime_prime.resize(this->number_of_entries_Newton);
  
  this->eta_Newton[0] = this->eta_initial;
  this->Newtonian_growth_factor_of_delta[0] = D;
  this->Newtonian_growth_factor_of_delta_prime[0] = D_prime;
  
  
  
  double e;
  double w;
  double de = (this->eta_final - this->eta_initial)/double(this->number_of_entries_Newton - 1);
  double e_i = this->eta_initial;
  double e_f = this->eta_initial + de;
  double D0;
  double y[2] = { this->Newtonian_growth_factor_of_delta[0], this->Newtonian_growth_factor_of_delta_prime[0]};
    
  gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, 1e-6, 1e-6, 0.0);

  for (int i = 1; i < this->number_of_entries_Newton; i++){
    
    
    this->eta_Newton[i] = e_f;
    int status = gsl_odeiv2_driver_apply(d, &e_i, e_f, y);
    
    if (status != GSL_SUCCESS){
      printf ("error, return value=%d\n", status);
      break;
    }
    
    
    double scale = this->universe->a_at_eta(e_f);
    double H = this->universe->H_at_eta(e_f);
    double H_prime = this->universe->H_prime_at_eta(e_f);
    double Om_m = this->cosmology.Omega_m;
    
    this->Newtonian_growth_factor_of_delta[i] = y[0];
    this->Newtonian_growth_factor_of_delta_prime[i] = y[1];
    this->Newtonian_growth_factor_of_delta_prime_prime[i] = -H*y[1]+3.0/2.0*Om_m/scale*y[0];
    this->Newtonian_growth_factor_of_delta_prime_prime_prime[i] = -H_prime*y[1]-H*this->Newtonian_growth_factor_of_delta_prime_prime[i]-3.0/2.0*H*Om_m/scale*y[0]+3.0/2.0*Om_m/scale*y[1];
    e_f += de;
  }

  gsl_odeiv2_driver_free(d);
  
  e = this->universe->eta_at_a(1.0);
  
  D0 = interpolate_neville_aitken(e, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  
  
  for (int i = 0; i < this->number_of_entries_Newton; i++){
    this->Newtonian_growth_factor_of_delta[i] /= D0;
    this->Newtonian_growth_factor_of_delta_prime[i] /= D0;
    this->Newtonian_growth_factor_of_delta_prime_prime[i] /= D0;
    this->Newtonian_growth_factor_of_delta_prime_prime_prime[i] /= D0;
  }
  
  
  
  /*for (int i = 1; i < this->number_of_entries_Newton; i++){
    w = this->universe->eta_at_a(1.0) - this->eta_Newton[i];
    cout << w << setw(20);
    cout << this->Newtonian_growth_factor_of_delta[i] << setw(20);
    cout << this->Newtonian_growth_factor_of_delta[i]*pow(w, 1.5) << setw(20);
    double w2 = this->universe->eta_at_a(1.0) - this->eta_Newton[i-1];
    cout << this->Newtonian_growth_factor_of_delta[i]*pow(w, 1.5)-this->Newtonian_growth_factor_of_delta[i-1]*pow(w2, 1.5) << setw(20);
    cout << (1.0 - w)*pow(w, 1.5)-(1.0 - w2)*pow(w2, 1.5) << '\n';
  }*/

}


/*******************************************************************************************************************************************************
 * 3.1 initialize_up_to_second_order_growth_factor_of_delta
 * Description:
 *
 * Arguments:
 * 
 * Comments:
 * - In this method there are still NO radiation inhomogeneities includes. This would also require to take into account
 *   entropy perturbation!
 * 
*******************************************************************************************************************************************************/

void Matter::initialize_up_to_second_order_growth_factor_of_delta(double D, double D_prime){

  integration_parameters params;
  params.pointer_to_Matter = this;
  params.pointer_to_Universe = this->universe;
  params.Omega_m = this->cosmology.Omega_m;
  integration_parameters * pointer_to_params = &params;
  
  gsl_odeiv2_system sys = {growth_factor_to_second_order_gsl, growth_factor_to_second_order_gsl_jac, 2, (void *) pointer_to_params};

  
  this->Newtonian_growth_factor_second_order.resize(this->number_of_entries_Newton);
  this->Newtonian_growth_factor_second_order_prime.resize(this->number_of_entries_Newton);
  
  this->Newtonian_growth_factor_second_order[0] = D;
  this->Newtonian_growth_factor_second_order_prime[0] = D_prime;
  
  
  
  double e;
  double de = (this->eta_final - this->eta_initial)/double(this->number_of_entries_Newton - 1);
  double e_i = this->eta_Newton[0];
  double e_f;
  double y[2] = { this->Newtonian_growth_factor_second_order[0], this->Newtonian_growth_factor_second_order_prime[0]};
    
  gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, 1e-6, 1e-6, 0.0);

  for (int i = 1; i < this->number_of_entries_Newton; i++){

    e_f = this->eta_Newton[i];
    int status = gsl_odeiv2_driver_apply(d, &e_i, e_f, y);
    
    if (status != GSL_SUCCESS){
      printf ("error, return value=%d\n", status);
      break;
    }
    
    this->Newtonian_growth_factor_second_order[i] = y[0];
    this->Newtonian_growth_factor_second_order_prime[i] = y[1];
    
    e_f += de;
  }

  gsl_odeiv2_driver_free(d);
  
  e = this->universe->eta_at_a(1.0);
  
  double D11 = interpolate_neville_aitken(e, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double D22 = interpolate_neville_aitken(e, &this->eta_Newton, &this->Newtonian_growth_factor_second_order, this->order);

  //cout << "HAHAHAHAHAHAHAHAHAHA\n";
  //cout << 2.0/7.0 << '\n';
  //cout << D22/D11/D11 << '\n';
}

/*******************************************************************************************************************************************************
 * 3.2 Newtonian_linear_power_spectrum
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

double Matter::Newtonian_linear_power_spectrum(double k, double e){

  
  double D = interpolate_neville_aitken(e, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double transfer_sq = this->transfer_function_at(k); transfer_sq *= transfer_sq;
  return D*D*pow(k/c_over_e5, this->cosmology.n_s)*transfer_sq*pow(this->cosmology.sigma_8, 2)/this->norm;
  //double ln_k = log(k);
  //return D*D*interpolate_neville_aitken(ln_k, &this->CAMB_ln_k_values, &this->CAMB_power_spectrum, this->order);
    
}

//marco
double Matter::Newtonian_linear_power_spectrum(double k){


  double transfer_sq = this->transfer_function_at(k); transfer_sq *= transfer_sq;
  return  pow(k, this->cosmology.n_s)*transfer_sq*pow(this->cosmology.sigma_8, 2);///this->norm;
  //double ln_k = log(k);
  //double ln_k = log(k);
  //return interpolate_neville_aitken(ln_k, &this->CAMB_ln_k_values, &this->CAMB_power_spectrum, this->order);

}




/*******************************************************************************************************************************************************
 * 5.12 set_spherical_collapse_evolution_of_delta
 * Description:
 * - for a number of initial values for delta, this function computes their evolution in the
 *   spherical collapse model
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

void Matter::set_spherical_collapse_evolution_of_delta(double z_min, double z_max, int n_time){

  double delta_min = -5.0;
  double delta_max = 1.4;
  double ddelta = 0.02;
  double delta = delta_min-ddelta;
  double eta_i = this->eta_Newton[0];
  double eta_min = this->universe->eta_at_a(1.0/(1+z_max));
  double eta_max = this->universe->eta_at_a(1.0/(1+z_min));
  double deta = (eta_max - eta_min)/double(n_time - 1);
  double D = this->Newtonian_growth_factor_of_delta[0];
  double H_initial = this->universe->H_at_eta(eta_i);
  
  int n = 1+int((delta_max - delta_min)/ddelta);
  
  
  this->delta_values.resize(n, 0.0);
  this->eta_NL.resize(n_time, 0.0);
  this->spherical_collapse_evolution_of_delta.resize(n, vector<double>(n_time, 0.0));
  this->spherical_collapse_evolution_of_delta_ddelta.resize(n, vector<double>(n_time, 0.0));
  this->F_prime_of_eta.resize(n_time, 0.0);
  this->F_prime_prime_of_eta.resize(n_time, 0.0);
  
  integration_parameters params;
  params.n_s = this->cosmology.n_s;
  params.Omega_m = this->cosmology.Omega_m;
  params.pointer_to_Universe = this->universe;
  params.pointer_to_Matter = this;
  params.top_hat_radius = 10.0;
  params.second_top_hat_radius = 10.0;
  integration_parameters * pointer_to_params = &params;
  
  //gsl_odeiv2_system sys = {spherical_collapse_and_dF_ddelta_gsl, spherical_collapse_gsl_and_dF_ddelta_jac, 4, (void *) pointer_to_params};
  //gsl_odeiv2_system sys_of_derivatives = {dF_ddelta_at_average_density_gsl, dF_ddelta_at_average_density_jac, 4, (void *) pointer_to_params};
  gsl_odeiv2_system sys = {cylindrical_collapse_and_dF_ddelta_gsl, cylindrical_collapse_gsl_and_dF_ddelta_jac, 4, (void *) pointer_to_params};
  gsl_odeiv2_system sys_of_derivatives = {dF_cylindrical_ddelta_at_average_density_gsl, dF_cylindrical_ddelta_at_average_density_jac, 4, (void *) pointer_to_params};
    
  for(int i = 0; i < n; i++){
    
    delta += ddelta;
    delta_values[i] = delta;
    
    gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rkf45, 1e-6, 1e-6, 0.0);
    
    eta_i = this->eta_Newton[0];
    //double y[2] = { delta*D, delta*D*H_initial };
    double y[4] = { delta*D, delta*D*H_initial, D, D*H_initial };
    double eta = eta_min;
    for (int j = 0; j < n_time; j++){
      this->eta_NL[j] = eta;
      int status = gsl_odeiv2_driver_apply(d, &eta_i, eta, y);
      if (status != GSL_SUCCESS){
        printf ("error, return value=%d\n", status);
        break;
      }
      this->spherical_collapse_evolution_of_delta[i][j] = y[0];
      this->spherical_collapse_evolution_of_delta_ddelta[i][j] = y[2];
      eta += deta;
    }

    gsl_odeiv2_driver_free(d);
    
  }
  
  //cout << delta_max << setw(20) << this->spherical_collapse_evolution_of_delta[n-1][n_time-1] << endl;
  
  double eta_0 = this->universe->eta_at_a(1.0);
  double y[4] = {D, D*H_initial, 0.0, 0.0};
  double eta = eta_min;
  eta_i = this->eta_Newton[0];
  gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new (&sys_of_derivatives, gsl_odeiv2_step_rkf45, 1e-6, 1e-6, 0.0);
  
  for (int j = 0; j < n_time; j++){
    int status = gsl_odeiv2_driver_apply(d, &eta_i, eta, y);
    if (status != GSL_SUCCESS){
      printf ("error, return value=%d\n", status);
      break;
    }
    this->F_prime_of_eta[j] = y[0];
    this->F_prime_prime_of_eta[j] = y[2];
    eta += deta;
    
  }
  gsl_odeiv2_driver_free(d);
  
    
  
  double eta_1 = this->universe->eta_at_a(0.1);
  double eta_2 = this->universe->eta_at_a(1.0);
  
  vector<double> delta_values_1(0, 0.0);
  vector<double> F_values_1(0, 0.0);
  vector<double> F_values_2(0, 0.0);
  vector<double> F_prime_values_1(0, 0.0);
    
  this->return_delta_NL_of_delta_L_and_dF_ddelta(eta_1, &delta_values_1, &F_values_1, &F_prime_values_1);
  this->return_delta_NL_of_delta_L_and_dF_ddelta(eta_2, &delta_values_1, &F_values_2, &F_prime_values_1);
  
  //for(int i = 0; i < delta_values_1.size(); i++){
  //  cout << delta_values_1[i] << setw(20) << F_values_1[i] << setw(20) << F_values_2[i] << '\n';
  //}
  
}

