
#include <gsl/gsl_poly.h>

// Here, threePt is <kappa delta^2>
double get_kappa0(double d0, double var, double cov, double threePt){
  
  //return max(0.1*cov/d0, cov*cov*(1.0 + var/d0/d0)/(threePt - 2.0*cov*var/d0));
  return cov*cov*(1.0 + var/d0/d0)/(threePt - 2.0*cov*var/d0);
  
}


// Here, threePt is <kappa^2 delta>
double get_var_kappa(double d0, double k0, double cov, double threePt){
  
  return (d0*threePt - cov*cov)/(cov*cov/(k0*k0) + 2.0*cov*d0/k0);
  
}

double get_cov_of_two_kappas(double d0, double k01, double k02, double dk1, double dk2, double dk1k2){
  
  return (dk1k2 - dk1*dk2/d0)/(dk1/k01 + dk2/k02 + dk1*dk2/(d0*k01*k02));
  
}



double f_of_y(double y, double twoPt, double threePt){
  
  return pow(y, 3.0)*threePt - 3.0*twoPt*twoPt*pow(y, 2.0)-pow(twoPt, 3.0);
  
}

double f_prime_of_y(double y, double twoPt, double threePt){
  
  return 3.0*pow(y, 2.0)*threePt - 6.0*twoPt*twoPt*y;
  
}

// Here, threePt is <delta^3>
double get_delta0(double twoPt, double threePt){
  
  double S3 = threePt/twoPt/twoPt;
  double x0, x1, x2;
  gsl_poly_solve_cubic(-3.0/S3, 0.0, -twoPt/S3, &x0, &x1, &x2);
  return x0;
  //return 3.0*pow(twoPt,2)/threePt;
  
  int step = 0;
  double y = 10.0;
  double f, f_prime;

  while(step < 100000){
    
    step ++;
    f = f_of_y(y, twoPt, threePt);
    f_prime = f_prime_of_y(y, twoPt, threePt);
    y = y - f/f_prime;
    
  }
  return y;
  
}


double expectation_of_kappa_given_delta(double d, double d0, double k0, double variance_d, double covariance){
  
  double COV = log(1.0 + covariance/d0/k0);
  double VAR = log(1.0 + variance_d/d0/d0);
  
  if(d <= -d0) return -k0;
  
  return k0*(exp(0.5*COV/VAR*(2.0*log(1.0+d/d0)+VAR-COV))-1.0);
  
}


double expectation_of_kappa_given_delta_as_function_of_kdd(double d, double d0, double kdd, double variance_d, double covariance){
  
  // If moments can be matched with a joint log-normal PDF
  if(kdd - 2.0*variance_d*covariance/d0 > 0.0){
    double k0 = covariance*covariance*(1.0 + variance_d/d0/d0)/(kdd - 2.0*covariance*variance_d/d0);
    double COV = log(1.0 + covariance/d0/k0);
    double VAR = log(1.0 + variance_d/d0/d0);
    if(d <= -d0) return -k0;
    return k0*(exp(0.5*COV/VAR*(2.0*log(1.0+d/d0)+VAR-COV))-1.0);
  }
  
  //otherwise just assume Gaussian PDF for kappa
  double VAR = log(1.0 + variance_d/d0/d0);
  if(d <= -d0) return d*covariance/variance_d;
  return covariance/(d0*VAR)*(log(1.0+d/d0)+VAR/2.0);
}


double expectation_of_kappa_given_delta_as_function_of_kdd(double d, double d0, double kdd, double variance_d, double covariance, double covariance_at_trough_radius){
  
  // If moments can be matched with a joint log-normal PDF
  if(kdd - 2.0*variance_d*covariance/d0 > 0.0){
    double k0 = covariance_at_trough_radius*covariance_at_trough_radius*(1.0 + variance_d/d0/d0)/(kdd - 2.0*covariance_at_trough_radius*variance_d/d0);
    double COV = log(1.0 + covariance/d0/k0);
    double VAR = log(1.0 + variance_d/d0/d0);
    if(d <= -d0) return -k0;
    return k0*(exp(0.5*COV/VAR*(2.0*log(1.0+d/d0)+VAR-COV))-1.0);
  }
  
  //otherwise just assume Gaussian PDF for kappa
  double VAR = log(1.0 + variance_d/d0/d0);
  if(d <= -d0) return d*covariance/variance_d;
  return covariance/(d0*VAR)*(log(1.0+d/d0)+VAR/2.0);
}

double PDF_of_delta(double d, double d0, double variance_d){

  double VAR = log(1.0 + variance_d/d0/d0);
  
  if(d <= -d0) return 0.0;
  
  return exp(-0.5*pow(log(1.0+d/d0) + 0.5*VAR, 2.0)/VAR)/sqrt(2.0*constants::pi*VAR)/(d0+d);

}

double PDF_of_delta_from_Gaussian_params(double d, double d0, double mu, double VAR){
  
  if(d <= -d0) return 0.0;
  
  return exp(-0.5*pow(log(1.0+d/d0) - mu, 2.0)/VAR)/sqrt(2.0*constants::pi*VAR)/(d0+d);

}

double PDF_of_Gaussian_field(double x, double mu, double VAR){
    
  return exp(-0.5*pow(x - mu, 2.0)/VAR)/sqrt(2.0*constants::pi*VAR);

}

double PDF_of_Gaussian_field_unnormalized(double x, double mu, double one_over_2VAR){
    
  return exp(-pow(x - mu, 2.0)*one_over_2VAR);

}

double dPDF_ddelta(double d, double d0, double variance_d){

  double VAR = log(1.0 + variance_d/d0/d0);
  double mu = log(d0) - 0.5*VAR;
  double G = log(d+d0);
  double dG_dd = 1.0/(d+d0);
  
  if(d <= -d0) return 0.0;
  
  return exp(-0.5*pow(G - mu, 2.0)/VAR)/sqrt(2.0*constants::pi*VAR)*pow(dG_dd, 2)*((G - mu)/VAR - 1.0);

}


double dg_given_dm_equals_0(double r, double bias, double variance, double delta_m0){
  
  double var_Gauss = log(1.0+variance/delta_m0/delta_m0);
  double rho = log(1.0+r*variance/delta_m0/delta_m0)/var_Gauss;
  
  double dg_given_nm = exp(0.5*var_Gauss*rho*(1.0 - rho));
  
  return bias*delta_m0*(dg_given_nm - 1.0);
    
}

double dg_given_dm_equals_0_deriv(double r, double bias, double variance, double delta_m0){
  
  double var_Gauss = log(1.0+variance/delta_m0/delta_m0);
  double rho = log(1.0+r*variance/delta_m0/delta_m0)/var_Gauss;
  
  double dg_given_nm = exp(0.5*var_Gauss*rho*(1.0 - rho));
  //cout << rho << '\nt';
  //cout << dg_given_nm << '\t';
  //cout << rho*dg_given_nm << '\n';
  return bias*rho*dg_given_nm;
    
}



double Var_ng_given_dm_equals_0(double r, double bias, double variance, double delta_m0){
  
  double var_Gauss = log(1.0+variance/delta_m0/delta_m0);
  double rho = log(1.0+r*variance/delta_m0/delta_m0)/var_Gauss;
  
  return (exp(var_Gauss*(1.0-rho*rho))-1.0)*exp(var_Gauss*rho*(1.0 - rho));
    
}

double Var_ng_given_dm_equals_0_deriv(double r, double bias, double variance, double delta_m0){
  
  double var_Gauss = log(1.0+variance/delta_m0/delta_m0);
  double rho = log(1.0+r*variance/delta_m0/delta_m0)/var_Gauss;
  
  return 2.0*rho/delta_m0*(exp(var_Gauss*(1.0-rho*rho))-1.0)*exp(var_Gauss*rho*(1.0 - rho));
    
}

double return_alpha_0(double r, double bias, double N_bar, double variance, double delta_m0){
  
  double dg_given_dm = dg_given_dm_equals_0(r, bias, variance, delta_m0);
  double Var_dg_given_dm = Var_ng_given_dm_equals_0(r, bias, variance, delta_m0);
  
  return 1.0+N_bar*bias*bias*delta_m0*delta_m0*Var_dg_given_dm/(1.0+dg_given_dm);
  
}

double return_alpha_1(double r, double bias, double N_bar, double variance, double delta_m0){
  
  double dg_given_dm = dg_given_dm_equals_0(r, bias, variance, delta_m0);
  double dg_given_dm_deriv = dg_given_dm_equals_0_deriv(r, bias, variance, delta_m0);
  double Var_dg_given_dm = Var_ng_given_dm_equals_0(r, bias, variance, delta_m0);
  double Var_dg_given_dm_deriv = Var_ng_given_dm_equals_0_deriv(r, bias, variance, delta_m0);
  
  return N_bar*bias*bias*delta_m0*delta_m0*Var_dg_given_dm_deriv/(1.0+dg_given_dm) - N_bar*bias*bias*delta_m0*delta_m0*Var_dg_given_dm/(1.0+dg_given_dm)/(1.0+dg_given_dm)*dg_given_dm_deriv;
  
}

