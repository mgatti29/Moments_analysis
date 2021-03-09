#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <cstring>


namespace constants{

/***** Enums *****/

enum SHOT_NOISE_MODUS {BA0A1, BR, INVALID_SHOT_NOISE};
enum PDF_MODUS {BERNARDEAU, LOGNORMAL, LOGNORMAL_FIX_D0, GAUSSIAN, INVALID_PDF}; 
enum BINNING {LIN, LOG}; 

const char *SHOT_NOISE_types[] = {"BA0A1", "BR", "INVALID_SHOTNOISE"};
const char *PDF_types[] = {"BERNARDEAU", "LOGNORMAL", "LOGNORMAL_FIX_D0", "GAUSSIAN", "INVALID_PDF"};
const char *BINNING_types[] = {"LIN", "LOG"};
  

/***** General constants *****/

static double eulers_constant = exp(1.0);
static double pi = 3.14159265;
static double pi2 = 2.0*3.14159265;
static double pi_sq = pi*pi;
static double sqrt2_plus_1 = 2.414213562373095;
static double rad = pi/(3600*180);
static double c_in_si =  2.99792458e8;
static double c_over_e5 = c_in_si/1.0e5;
static double c_over_e5_sq = c_over_e5*c_over_e5;

  /*
    TOM WAS HERE
    - c_over_e5 is divided into many things. To get rid of divisions we can calculate the inverse ahead of time and just multiply this on.
   */

static double inverse_c_over_e5 = 1.0/c_over_e5;
static double arcmin = 2.908882087e-4;

/***** epsilon to judge whether something is 0 (zero) *****/

static double eps_from_0 = 1.0e-10;

/***** Variables controlling the precision finding roots by Newton's method or my nested intervals *****/

static double eps_Newton = 1.0e-5;
static double eps_nested = 1.0e-5;

/***** Variables controlling the precision of integrating differential equations in Universe.cpp and Matter.cpp *****/

static double background_max_dev = 0.5e-14;
static double growth_factor_max_dev = 0.5e-14;


/***** Max ell cut off for 2D power spectra *****/

static int ell_max = 10000; //10000;
/* FLASK ell_max: */
//static int ell_max = 4096;
//static int ell_max = 2*4096+1;


/***** Variables controlling the of interpolating polynomials *****/

static int coeff_order = 10;

/***** Variables controlling the precision of all integrals over the 3D-power spectrum in Matter.cpp *****/

// values of wave numbers in h/Mpc
static int number_of_k = 10000;
//static int number_of_k = 1024;
static double minimal_wave_number = 3.336e-6;
static double maximal_wave_number = 3336.0;//333.6;
static double sig_sq_precision = 0.0001;



//************
static int len_ww =1000;
};





namespace overloads{

  std::string to_string(int val) {
    return std::to_string((long long) (val));
  }

  std::string to_string(double val) {
    return std::to_string((long double) (val));
  }

}
