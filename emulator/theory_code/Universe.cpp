#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
//#include <armadillo>

//#include "H5Cpp.h"
#include "cosmology_utils.h"
#include "Universe.h"
#include "Universe_integrands.h"
 
using namespace std;
using namespace constants;
//using namespace arma;
//using namespace H5;

/* _____ Content Universe.cpp
 * 
 * _____ 1. Initialization
 * 
 * ..... 1.1 Constructor
 * ..... 1.2 Destructor
 * ..... 1.3 estimate_initial_step_size
 * ..... 1.4 set_background_cosmology
 * 
 * _____ 2. Cosmological Functions
 * 
 * ..... 2.1 hubble_start
 * ..... 2.2 hubble_prime
 * 
 * _____ 3. Return Values
 * 
 * ..... 3.1  return_eta                   
 * ..... 3.2  return_number_of_entries     
 * ..... 3.3  return_cosmology             
 * ..... 3.4  f_k                          
 * ..... 3.5  rho_m_of_a                   
 * ..... 3.6  rho_r_of_a                   
 * ..... 3.7  rho_L_of_a                   
 * ..... 3.8  w_L_of_a                     
 * ..... 3.9  a_at_eta                     
 * ..... 3.10 H_at_eta                     
 * ..... 3.11 H_prime_at_eta               
 * ..... 3.12 eta_at_a                     
 * 
 * _____ 4. Output and Checks
 * 
 * ..... 4.1 print_background_cosmology
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
 * ..... 1.3 estimate_initial_step_size    *
 * ..... 1.4 set_background_cosmology      *
 *                                         *
 *******************************************
 *******************************************/


/*******************************************************************************************************************************************************
 * 1.1 Constructor
 * Description:
 *  - Constructor of class Universe
 * Arguments:
 *  - cosmo: struct containing cosmological parameters
 *  - t_start: initial value of time parameter
 *  - a_min: minimal scale faction (= initial scale factor for expanding universe)
 *  - a_max: maximal scale faction
 *  - expand_or_collapse: 1 == expanding universe, 0 == collapsing universe
 * 
*******************************************************************************************************************************************************/


Universe::Universe(cosmological_model cosmo, double t_start, double a_min, double a_max, int expand_or_collapse){
  
  this->t_initial = t_start;
  this->eta_initial = 0;                                // eta_initial is always assumed to be 0, other values are only needed in the consistency check method
  this->expansion_or_collapse = expand_or_collapse;
  this->order = 5;
  
  if(expand_or_collapse){
    this->a_initial = a_min;
    this->a_final = a_max;
  }
  else{
    this->a_initial = a_max;
    this->a_final = a_min;
  }

  if(cosmo.Omega_r<0.) {
     cerr << "Omega_r=" << cosmo.Omega_r << endl;
     assert(1==2);
  }
  if(cosmo.theta_27<0.) {
     cerr << "theta_27=" << cosmo.theta_27 << endl;
     assert(1==2);
  }
  if(cosmo.Omega_m<0.) {
     cerr << "Omega_m=" << cosmo.Omega_m << endl;
     assert(1==2);
  }
  if(cosmo.Omega_L<0.) {
     cerr << "Omega_L=" << cosmo.Omega_L << endl;
     assert(1==2);
  }
  if(cosmo.Omega_b<0.) {
     cerr << "Omega_b=" << cosmo.Omega_b << endl;
     assert(1==2);
  }
  if(cosmo.Omega_b>cosmo.Omega_m) {
     cerr << "Omega_b=" << cosmo.Omega_b << endl;
     cerr << "Omega_m=" << cosmo.Omega_m << endl;
     assert(1==2);
  }
  if(cosmo.h_100<=0.) {
     cerr << "h_100=" << cosmo.h_100 << endl;
     assert(1==2);
  }
  if(cosmo.sigma_8<0.) {
     cerr << "sigma_8=" << cosmo.sigma_8 << endl;
     assert(1==2);
  }

  this->cosmology = cosmo;
  
  this->set_background_cosmology();

}

/*******************************************************************************************************************************************************
 * 1.2 Destructor
 * Description:
 * 
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

Universe::~Universe(){
  //cout << "\nThe Universe was destroyed - probably by a protouniverse intruding our universe through a gap in space time.\n(see Star Trek: Deep Space 9)\n\n";
}

/*******************************************************************************************************************************************************
 * 1.3 estimate_initial_step_size
 * Description:
 *  - Estimates initial step size (in conformal time) that will be used for integrating the Friedmann equations.
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

double Universe::estimate_initial_step_size(){
  
  double a_i = this->a_initial;
  double a_f = this->a_final;

  double da = (a_f-a_i)/(1.*len_ww);
  double H_i = this->hubble_start(a_i);
  
  return da/(a_i*H_i);
  
}

/*******************************************************************************************************************************************************
 * 1.4 set_background_cosmology
 * Description:
 *  - Integrates Friedmann equations to give
 *    a(eta)
 *    H(eta)
 *    H'(eta)
 *    t(eta)
 *    where eta is conformal time and H is the conformal expansion rate a'/a.
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

void Universe::set_background_cosmology(){
  
  integration_parameters_Universe params;
  params.pointer_to_Universe = this;
  integration_parameters_Universe * pointer_to_params = &params;
  
  gsl_odeiv2_system sys = {scale_factor_gsl, scale_factor_gsl_jac, 2, (void *) pointer_to_params};

  this->set_number_of_entries(len_ww);
	
  this->a.resize(this->number_of_entries);
  this->H.resize(this->number_of_entries);
  this->eta.resize(this->number_of_entries);
  this->t.resize(this->number_of_entries);
  this->H_prime.resize(this->number_of_entries);
	
	this->a[0] = a_initial;
	this->eta[0] = this->eta_initial;
	this->H[0] = double(this->expansion_or_collapse)*this->hubble_start(a_initial);
	
	
  double cosmo_t = this->t_initial;
	double da = (1.0 - a_initial)/double(this->number_of_entries - 1);
	double a_i = a_initial;
	double a_f = a_initial + da;
	double y[2] = { this->eta[0], this->H[0]};
    
  gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, 1e-6, 1e-6, 0.0);

	for (int i = 1; i < this->number_of_entries; i++){

    this->a[i] = a_f;
    int status = gsl_odeiv2_driver_apply(d, &a_i, a_f, y);
		
    if (status != GSL_SUCCESS){
	  	printf ("error, return value=%d\n", status);
	  	break;
		}
		this->eta[i] = y[0];
		this->H[i] = y[1];
    this->H_prime[i] = this->hubble_prime(this->a[i]);
    this->t[i] = cosmo_t;
    cosmo_t += (this->eta[i]-this->eta[i-1])*this->a[i];
		a_f += da;
  }

  gsl_odeiv2_driver_free(d);
}


/***********************************************
 ***********************************************
 **________ 2. COSMOLOGICAL FUNCTIONS ________**
 ***********************************************
 *********************************************** 
 *                                             *
 * ..... 2.1 hubble_start                      *
 * ..... 2.2 hubble_prime                      *
 *                                             *
 ***********************************************
 ***********************************************/

/*******************************************************************************************************************************************************
 * 2.1 hubble_start
 * Description:
 *  - Gives conformal expansion rate H at given scale factor. Within the code, this function is mainly used
 *    to set the initial value of H when integrating the Friedmann equations.
 * Arguments:
 *  - a_start: scale factor at which H shall be computed
 * 
*******************************************************************************************************************************************************/

double Universe::hubble_start(double a_start){
  
  double rho_m = rho_m_of_a(a_start);
  double rho_l = rho_L_of_a(a_start);
  double rho_r = rho_r_of_a(a_start);

  return sqrt(a_start*a_start*(rho_m+rho_r+rho_l)+this->cosmology.Omega_k);
  
}

/*******************************************************************************************************************************************************
 * 2.2 hubble_prime
 *  - Gives derivative of conformal expansion rate, H', at given scale factor. Within the code, this function
 *    is mainly used to set the initial value of H' when integrating the Friedmann equations.
 * Arguments:
 *  - a_start: scale factor at which H' shall be computed
 * 
*******************************************************************************************************************************************************/

double Universe::hubble_prime(double scale){
  
  double rho_m = rho_m_of_a(scale);
  double rho_l = rho_L_of_a(scale);
  double rho_r = rho_r_of_a(scale);
  double w = this->w_L_of_a(scale);

  return -0.5*(rho_m + 2*rho_r+(1.0+3.0*w)*rho_l)*scale*scale;
  
}


/*******************************************
 *******************************************
 **__________ 3. RETURN VALUES ___________**
 *******************************************
 ******************************************* 
 *                                         *
 * ..... 3.1  return_eta                   *
 * ..... 3.2  return_number_of_entries     *
 * ..... 3.3  return_cosmology             *
 * ..... 3.4  f_k                          *
 * ..... 3.5  rho_m_of_a                   *
 * ..... 3.6  rho_r_of_a                   *
 * ..... 3.7  rho_L_of_a                   *
 * ..... 3.8  w_L_of_a                     *
 * ..... 3.9  a_at_eta                     *
 * ..... 3.10 H_at_eta                     *
 * ..... 3.11 H_prime_at_eta               *
 * ..... 3.12 eta_at_a                     *
 *                                         *
 *******************************************
 *******************************************/


/*******************************************************************************************************************************************************
 * 3.1 return_eta
 * Description:
 *  Returns i-th element of the array containing the values of conformal time.
 * Arguments:
 *  - i: index, at which array element shall be returned
 * 
*******************************************************************************************************************************************************/

double Universe::return_eta(int i){
  
  if(i > this->a.size()) cerr << "Index for vector \"eta\" is too high in method \"return_eta\"!\n";
  
  return this->eta[i];
  
}

/*******************************************************************************************************************************************************
 * 3.2 return_number_of_entries
 * Description:
 *  - Returns number of elements stored in the arrays containing the expansion history (eta, a, t, H, H').
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

int Universe::return_number_of_entries(){
  
  return this->number_of_entries;
  
}

/*******************************************************************************************************************************************************
 * 3.3 return_cosmology
 * Description:
 *  - Returns struct with cosmological parameters.
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

cosmological_model Universe::return_cosmology(){
  
  return this->cosmology;
  
}

/*******************************************************************************************************************************************************
 * 3.4 f_k
 * Description:
 *  - Co-moving angular diameter distance.
 * Arguments:
 *  - w: co-moving distance at which f_k shall be evaluated
 * 
*******************************************************************************************************************************************************/

double Universe::f_k(double w){
  double omega_0 = 1.0-this->cosmology.Omega_k;
  
  if(abs(1.0-omega_0) < eps_from_0)
    return w;

  if(omega_0 > 1.0)
    return sin(w);
  
  return sinh(w);

  
}


/*******************************************************************************************************************************************************
 * 2.5 rho_m_of_a
 * Description:
 *  - Matter density at scale factor a in units of present day critical density.
 * Arguments:
 *  - a: scale factor
 * 
*******************************************************************************************************************************************************/

double Universe::rho_m_of_a(double a){
  
  return this->cosmology.Omega_m/pow(a, 3.0);
  
}



/*******************************************************************************************************************************************************
 * 2.6 rho_r_of_a
 * Description:
 *  - Radiation density at scale factor a in units of present day critical density.
 * Arguments:
 *  - a: scale factor
 * 
*******************************************************************************************************************************************************/


double Universe::rho_r_of_a(double a){
  
  return this->cosmology.Omega_r/pow(a, 4.0);
  
}


/*******************************************************************************************************************************************************
 * 2.7 rho_L_of_a
 * Description:
 *  - Dark energy density at scale factor a in units of present day critical density.
 * Arguments:
 *  - a: scale factor
 * 
*******************************************************************************************************************************************************/


double Universe::rho_L_of_a(double a){
  
  double w = this->w_L_of_a(a);
  
  return this->cosmology.Omega_L/pow(a, 3.0*(1.0+w));
  
}


/*******************************************************************************************************************************************************
 * 2.8 w_L_of_a
 * Description:
 * - Equation of State parameter of Dark energy. 
 * Arguments:
 *  - a: scale factor
 * 
*******************************************************************************************************************************************************/

double Universe::w_L_of_a(double a){
  
  double w1 = this->cosmology.w1;
  if(w1 == 0.0)
    return this->cosmology.w0;
  
  return this->cosmology.w0 + this->cosmology.w1*(1.0-a); // Chevallier-Polarski-Linde
  //return this->cosmology.w0 + this->cosmology.w1*(1.0-a)/a; // Linear-redshif
  //return this->cosmology.w0 + this->cosmology.w1*(1.0-a)*a; // Jassal-Bagla-Padmanabhan
  
}




/*******************************************************************************************************************************************************
 * 3.9 a_at_eta
 * Description:
 *  - Gives value of scale factor at given conformal time.
 * Arguments:
 *  - e: conformal time
 * 
*******************************************************************************************************************************************************/

double Universe::a_at_eta(double e){
  
  return interpolate_neville_aitken(e, &this->eta, &this->a, this->order);
  
}

/*******************************************************************************************************************************************************
 * 3.10 H_at_eta
 * Description:
 *  - Gives value of conformal expansion rate at given conformal time.
 * Arguments:
 *  - e: conformal time
 * 
*******************************************************************************************************************************************************/

double Universe::H_at_eta(double e){
  
  return interpolate_neville_aitken(e, &this->eta, &this->H, this->order);
  
}

/*******************************************************************************************************************************************************
 * 3.11 H_prime_at_eta
 * Description:
 *  - Gives derivative of conformal expansion rate at given conformal time.
 * Arguments:
 *  - e: conformal time
 * 
*******************************************************************************************************************************************************/

double Universe::H_prime_at_eta(double e){
  
  return interpolate_neville_aitken(e, &this->eta, &this->H_prime, this->order);
  
}

/*******************************************************************************************************************************************************
 * 3.12 eta_at_a
 * Description:
 *  - Gives value of conformal time at given scale factor.
 * Arguments:
 *  - a: scale factor
 * 
*******************************************************************************************************************************************************/

double Universe::eta_at_a(double a){
  
  return interpolate_neville_aitken(a, &this->a, &this->eta, this->order);
  
}


/*******************************************
 *******************************************
 **________ 4. OUTPUT AND CHECKS _________**
 *******************************************
 ******************************************* 
 *                                         *
 * ..... 4.1 print_background_cosmology    *
 *                                         *
 *******************************************
 *******************************************/

/*******************************************************************************************************************************************************
 * 4.1 print_background_cosmology
 * Description:
 *  - Prints table with expansion history of the universe.
 * Arguments:
 *  - file_name: file name to which the table shall be output
 * 
*******************************************************************************************************************************************************/

void Universe::print_background_cosmology(string file_name){
  
  double Om_m = this->cosmology.Omega_m;
  double Om_l = this->cosmology.Omega_L;
  double Om_r = this->cosmology.Omega_r;
  fstream cosmo_stream;
  
  remove(file_name.c_str());
  FILE * F = fopen(file_name.c_str(), "w");
  fclose(F);
  cosmo_stream.open(file_name.c_str());
  
  cosmo_stream << "Cosmological Parameters: Om_m = " << Om_m << ", Om_l = " << Om_l << ", Om_r = " << Om_r << '\n';
  cosmo_stream << "t" << setw(20) << "eta(t)" << setw(20) << "w(eta)" << setw(20) << "z(eta)" << setw(20) << "a(eta)" << setw(20) << "H(eta)" << setw(20) << "H_prime(eta)\n";
  for(int i = 0; i < this->a.size(); i++){
    cosmo_stream << this->t[i];
    cosmo_stream << setw(20) << this->eta[i];
    cosmo_stream << setw(20) << this->eta_at_a(1.0)-this->eta[i];
    cosmo_stream << setw(20) << 1.0/this->a[i] - 1.0;
    cosmo_stream << setw(20) << this->a[i];
    cosmo_stream << setw(20) << this->H[i];
    cosmo_stream << setw(20) << this->H_prime[i] << '\n';
  }
    
  cosmo_stream.close();
  
}




  




