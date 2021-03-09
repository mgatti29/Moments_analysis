#include <stdio.h>
#include <gsl/gsl_randist.h>


struct integration_parameters{
  
  double top_hat_radius;
  double second_top_hat_radius;
  double n_s;
  double Omega_m;
  
  Universe* pointer_to_Universe;
  Matter* pointer_to_Matter;
  
};


double norm_derivs_gsl(double lnk, void *params){
 
  integration_parameters *integration_params = (integration_parameters *) params;
  Matter* pointer_to_Matter = integration_params->pointer_to_Matter;
  
  double k = exp(lnk);
  double index = 3.0+integration_params->n_s;
  double WR = w_R(k, integration_params->top_hat_radius);
  double T_sq = pointer_to_Matter->transfer_function_at(k*c_over_e5); T_sq *= T_sq;
  
  return pow(k,index)*T_sq*WR*WR;
  
}


double halofit_sig_sq_gsl(double lnk, void *params){
 
  integration_parameters *integration_params = (integration_parameters *) params;
  Matter* pointer_to_Matter = integration_params->pointer_to_Matter;
  
  double index = 3.0+integration_params->n_s;
  
  double k = exp(lnk);
  double y_sq = pow(integration_params->top_hat_radius*k, 2);
  double T_sq = pointer_to_Matter->transfer_function_at(k*c_over_e5); T_sq *= T_sq;
  

  return pow(k,index)*T_sq*exp(-y_sq);
  
}


double halofit_C_gsl(double lnk, void *params){
  
  integration_parameters *integration_params = (integration_parameters *) params;
  Matter* pointer_to_Matter = integration_params->pointer_to_Matter;
  
  double index = 3.0+integration_params->n_s;
  double k = exp(lnk);
  double k_sq = k*k;
  double y_sq = pow(integration_params->top_hat_radius*k, 2);
  double T_sq = pointer_to_Matter->transfer_function_at(k*c_over_e5); T_sq *= T_sq;
  double integrand = pow(k,index)*T_sq*exp(-y_sq)*y_sq;
    
  return integrand;
  
}


double halofit_n_gsl(double lnk, void *params){
  
  integration_parameters *integration_params = (integration_parameters *) params;
  Matter* pointer_to_Matter = integration_params->pointer_to_Matter;
  
  double index = 3.0+integration_params->n_s;
  double k = exp(lnk);
  double k_sq = k*k;
  double y_sq = pow(integration_params->top_hat_radius*k, 2);
  double T_sq = pointer_to_Matter->transfer_function_at(k*c_over_e5); T_sq *= T_sq;
  double integrand = pow(k,index)*T_sq*exp(-y_sq)*y_sq;
    
  return (1-y_sq)*integrand;
  
}

double norm_derivs_2D_gsl(double lnk, void *params){
 
  integration_parameters *integration_params = (integration_parameters *) params;
  Matter* pointer_to_Matter = integration_params->pointer_to_Matter;
  
  double k = exp(lnk);
  double index = 2.0;
  double WR = w_R_2D(k, integration_params->top_hat_radius);
  double P = pointer_to_Matter->current_P_L_at(lnk);
  
  return pow(k,index)*P*WR*WR;
  
}

double covariance_derivs_2D_gsl(double lnk, void *params){
 
  integration_parameters *integration_params = (integration_parameters *) params;
  Matter* pointer_to_Matter = integration_params->pointer_to_Matter;
  
  double k = exp(lnk);
  double index = 2.0;
  double WR_1 = w_R_2D(k, integration_params->top_hat_radius);
  double WR_2 = w_R_2D(k, integration_params->second_top_hat_radius);
  double P = pointer_to_Matter->current_P_L_at(lnk);
  
  return pow(k,index)*P*WR_1*WR_2;
  
}



double int_gsl_integrate_medium_precision(double (*func)(double, void*),void *arg,double a, double b, double *error, int niter)
{
  double res, err;
  gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(niter);
  gsl_function F;
  F.function = func;
  F.params  = arg;
  gsl_integration_cquad(&F,a,b,0,1.0e-6,w,&res,&err,0);
  if(NULL!=error)
    *error=err;
  gsl_integration_cquad_workspace_free(w);
  return res;
}


double int_gsl_integrate_low_precision(double (*func)(double, void*),void *arg,double a, double b, double *error, int niter)
{
  double res, err;
  gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(niter);
  gsl_function F;
  F.function = func;
  F.params  = arg;
  gsl_integration_cquad(&F,a,b,0,1.0e-3,w,&res,&err,0);
  if(NULL!=error)
    *error=err;
  gsl_integration_cquad_workspace_free(w);
  return res;
}


int growth_factor_gsl(double e, const double D[], double dDde[], void *params){
  
  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double Om_m = integration_params->Omega_m;
  
  double D0 = D[0];
  double D1 = D[1];

  double scale = pointer_to_Universe->a_at_eta(e);
  double H = pointer_to_Universe->H_at_eta(e);

  dDde[0] = D1;
  dDde[1] = -H*D1+3.0/2.0*Om_m/scale*D0;
  
  return GSL_SUCCESS;
}

int growth_factor_to_second_order_gsl(double e, const double D[], double dDde[], void *params){
  
  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  Matter* pointer_to_Matter = integration_params->pointer_to_Matter;
  
  double Om_m = integration_params->Omega_m;
  
  double D0 = D[0];
  double D1 = D[1];

  double scale = pointer_to_Universe->a_at_eta(e);
  double H = pointer_to_Universe->H_at_eta(e);
  double D_lin_prime = pointer_to_Matter->return_Delta_prime_of_eta(e);

  dDde[0] = D1;
  dDde[1] = -H*D1+3.0/2.0*Om_m/scale*D0 + D_lin_prime*D_lin_prime;
  
  return GSL_SUCCESS;
  
}


int growth_factor_gsl_jac(double e, const double D[], double *dfdD, double dDde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double Om_m = integration_params->Omega_m;
  
  double D0 = D[0];
  double D1 = D[1];

  double scale = pointer_to_Universe->a_at_eta(e);
  double H = pointer_to_Universe->H_at_eta(e);

  dDde[0] = D1;
  dDde[1] = -H*D1+3.0/2.0*Om_m/scale*D0;
  
  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdD, 2, 2);
  gsl_matrix * m = &dfdy_mat.matrix;
  
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 1.0);
  
  gsl_matrix_set (m, 1, 0, 3.0/2.0*Om_m/scale);
  gsl_matrix_set (m, 1, 1, -H);

  return GSL_SUCCESS;
}


int growth_factor_to_second_order_gsl_jac(double e, const double D[], double *dfdD, double dDde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double Om_m = integration_params->Omega_m;
  
  double D0 = D[0];
  double D1 = D[1];

  double scale = pointer_to_Universe->a_at_eta(e);
  double H = pointer_to_Universe->H_at_eta(e);

  dDde[0] = D1;
  dDde[1] = -H*D1+3.0/2.0*Om_m/scale*D0;
  
  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdD, 2, 2);
  gsl_matrix * m = &dfdy_mat.matrix;
  
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 1.0);
  
  gsl_matrix_set (m, 1, 0, 3.0/2.0*Om_m/scale);
  gsl_matrix_set (m, 1, 1, -H);

  return GSL_SUCCESS;
}




int spherical_collapse_gsl(double e, const double y[], double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double delta = y[0];
  double delta_prime = y[1];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  dfde[0] = delta_prime;
  dfde[1] = 1.5*Omega_m/a*delta*(1+delta) + 4.0/3.0*pow(delta_prime, 2.0)/(1+delta) - hubble*delta_prime;
  return GSL_SUCCESS;
}


int spherical_collapse_gsl_jac(double e, const double y[], double *dfdy, double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;

  double delta = y[0];
  double delta_prime = y[1];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double hubble_prime = pointer_to_Universe->H_prime_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 2, 2);
  gsl_matrix * m = &dfdy_mat.matrix; 
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 1.0);
  gsl_matrix_set (m, 1, 0, 1.5*Omega_m/a*(1+2.0*delta) - 4.0/3.0*pow(delta_prime/(1+delta), 2.0));
  gsl_matrix_set (m, 1, 1, 8.0/3.0*delta_prime/(1+delta) - hubble);

  return GSL_SUCCESS;
}


int spherical_collapse_and_dF_ddelta_gsl(double e, const double y[], double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double F = y[0];
  double F_prime = y[1];
  double dF_ddelta = y[2];
  double dF_ddelta_prime = y[3];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  dfde[0] = F_prime;
  dfde[1] = 1.5*Omega_m/a*F*(1.0+F) + 4.0/3.0*pow(F_prime, 2)/(1+F) - hubble*F_prime;
  dfde[2] = dF_ddelta_prime;
  dfde[3] = 1.5*Omega_m/a*(dF_ddelta*(1.0+2.0*F)) + 8.0/3.0*F_prime*dF_ddelta_prime/(1+F) - 4.0/3.0*pow(F_prime, 2)/pow(1+F, 2)*dF_ddelta - hubble*dF_ddelta_prime;
  return GSL_SUCCESS;
}


int spherical_collapse_gsl_and_dF_ddelta_jac(double e, const double y[], double *dfdy, double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;

  double F = y[0];
  double F_prime = y[1];
  double dF_ddelta = y[2];
  double dF_ddelta_prime = y[3];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 4, 4);
  gsl_matrix * m = &dfdy_mat.matrix; 
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 1.0);
  gsl_matrix_set (m, 0, 2, 0.0);
  gsl_matrix_set (m, 0, 3, 0.0);
  
  gsl_matrix_set (m, 1, 0, 1.5*Omega_m/a*(1+2.0*F) - 4.0/3.0*pow(F_prime/(1+F), 2.0));
  gsl_matrix_set (m, 1, 1, 8.0/3.0*F_prime/(1+F) - hubble);
  gsl_matrix_set (m, 1, 2, 0.0);
  gsl_matrix_set (m, 1, 3, 0.0);
   
  gsl_matrix_set (m, 2, 0, 0.0);
  gsl_matrix_set (m, 2, 1, 0.0);
  gsl_matrix_set (m, 2, 2, 0.0);
  gsl_matrix_set (m, 2, 3, 1.0);  
   
  gsl_matrix_set (m, 3, 0, 3.0*Omega_m/a*dF_ddelta - 8.0/3.0/pow(1+F, 2)*F_prime*(dF_ddelta_prime - F_prime/(1+F)*dF_ddelta));
  gsl_matrix_set (m, 3, 1, 8.0/3.0/(1+F)*(dF_ddelta_prime + F_prime/(1+F)*dF_ddelta));
  gsl_matrix_set (m, 3, 2, 1.5*Omega_m/a*(1.0+2.0*F) - 4.0/3.0*pow(F_prime, 2)/pow(1+F, 2));
  gsl_matrix_set (m, 3, 3, 8.0/3.0*F_prime/(1+F) - hubble);
  
  return GSL_SUCCESS;
}




int dF_ddelta_at_average_density_gsl(double e, const double y[], double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double dF_ddelta = y[0];
  double dF_ddelta_prime = y[1];
  double d2F_ddelta2 = y[2];
  double d2F_ddelta2_prime = y[3];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  dfde[0] = dF_ddelta_prime;
  dfde[1] = 1.5*Omega_m/a*dF_ddelta - hubble*dF_ddelta_prime;
  dfde[2] = d2F_ddelta2_prime;
  dfde[3] = 1.5*Omega_m/a*(d2F_ddelta2 + 2.0*pow(dF_ddelta, 2)) + 8.0/3.0*pow(dF_ddelta_prime, 2) - hubble*d2F_ddelta2_prime;
  return GSL_SUCCESS;
}


int dF_ddelta_at_average_density_jac(double e, const double y[], double *dfdy, double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;

  double dF_ddelta = y[0];
  double dF_ddelta_prime = y[1];
  double d2F_ddelta2 = y[2];
  double d2F_ddelta2_prime = y[3];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 4, 4);
  gsl_matrix * m = &dfdy_mat.matrix; 
  
  
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 1.0);
  gsl_matrix_set (m, 0, 2, 0.0);
  gsl_matrix_set (m, 0, 3, 0.0);

  gsl_matrix_set (m, 1, 0, 1.5*Omega_m/a);
  gsl_matrix_set (m, 1, 1, -hubble);
  gsl_matrix_set (m, 1, 2, 0.0);
  gsl_matrix_set (m, 1, 3, 0.0);
   
  gsl_matrix_set (m, 2, 0, 0.0);
  gsl_matrix_set (m, 2, 1, 0.0);
  gsl_matrix_set (m, 2, 2, 0.0);
  gsl_matrix_set (m, 2, 3, 1.0);

  gsl_matrix_set (m, 3, 0, 6.0*Omega_m/a*dF_ddelta);
  gsl_matrix_set (m, 3, 1, 16.0/3.0*dF_ddelta_prime);
  gsl_matrix_set (m, 3, 2, 1.5*Omega_m/a);
  gsl_matrix_set (m, 3, 3, -hubble);
  
  return GSL_SUCCESS;
}






/*
 * 
 * Cylindrical collapse
 * 
 * 
 * 
 */





int cylindrical_collapse_gsl(double e, const double y[], double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double delta = y[0];
  double delta_prime = y[1];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  dfde[0] = delta_prime;
  dfde[1] = 1.5*Omega_m/a*delta*(1+delta) + 3.0/2.0*pow(delta_prime, 2.0)/(1+delta) - hubble*delta_prime;
  return GSL_SUCCESS;
}


int cylindrical_collapse_gsl_jac(double e, const double y[], double *dfdy, double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;

  double delta = y[0];
  double delta_prime = y[1];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double hubble_prime = pointer_to_Universe->H_prime_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 2, 2);
  gsl_matrix * m = &dfdy_mat.matrix; 
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 1.0);
  gsl_matrix_set (m, 1, 0, 1.5*Omega_m/a*(1+2.0*delta) - 3.0/2.0*pow(delta_prime/(1+delta), 2.0));
  gsl_matrix_set (m, 1, 1, 6.0/2.0*delta_prime/(1+delta) - hubble);

  return GSL_SUCCESS;
}


int cylindrical_collapse_and_dF_ddelta_gsl(double e, const double y[], double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double F = y[0];
  double F_prime = y[1];
  double dF_ddelta = y[2];
  double dF_ddelta_prime = y[3];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  dfde[0] = F_prime;
  dfde[1] = 1.5*Omega_m/a*F*(1.0+F) + 3.0/2.0*pow(F_prime, 2)/(1+F) - hubble*F_prime;
  dfde[2] = dF_ddelta_prime;
  dfde[3] = 1.5*Omega_m/a*(dF_ddelta*(1.0+2.0*F)) + 6.0/2.0*F_prime*dF_ddelta_prime/(1+F) - 3.0/2.0*pow(F_prime, 2)/pow(1+F, 2)*dF_ddelta - hubble*dF_ddelta_prime;
  return GSL_SUCCESS;
}


int cylindrical_collapse_gsl_and_dF_ddelta_jac(double e, const double y[], double *dfdy, double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;

  double F = y[0];
  double F_prime = y[1];
  double dF_ddelta = y[2];
  double dF_ddelta_prime = y[3];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 4, 4);
  gsl_matrix * m = &dfdy_mat.matrix; 
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 1.0);
  gsl_matrix_set (m, 0, 2, 0.0);
  gsl_matrix_set (m, 0, 3, 0.0);
  
  gsl_matrix_set (m, 1, 0, 1.5*Omega_m/a*(1+2.0*F) - 3.0/2.0*pow(F_prime/(1+F), 2.0));
  gsl_matrix_set (m, 1, 1, 6.0/2.0*F_prime/(1+F) - hubble);
  gsl_matrix_set (m, 1, 2, 0.0);
  gsl_matrix_set (m, 1, 3, 0.0);
   
  gsl_matrix_set (m, 2, 0, 0.0);
  gsl_matrix_set (m, 2, 1, 0.0);
  gsl_matrix_set (m, 2, 2, 0.0);
  gsl_matrix_set (m, 2, 3, 1.0);  
   
  gsl_matrix_set (m, 3, 0, 3.0*Omega_m/a*dF_ddelta - 6.0/2.0/pow(1+F, 2)*F_prime*(dF_ddelta_prime - F_prime/(1+F)*dF_ddelta));
  gsl_matrix_set (m, 3, 1, 6.0/2.0/(1+F)*(dF_ddelta_prime + F_prime/(1+F)*dF_ddelta));
  gsl_matrix_set (m, 3, 2, 1.5*Omega_m/a*(1.0+2.0*F) - 3.0/2.0*pow(F_prime, 2)/pow(1+F, 2));
  gsl_matrix_set (m, 3, 3, 6.0/2.0*F_prime/(1+F) - hubble);
  
  return GSL_SUCCESS;
}




int dF_cylindrical_ddelta_at_average_density_gsl(double e, const double y[], double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;
  
  double dF_ddelta = y[0];
  double dF_ddelta_prime = y[1];
  double d2F_ddelta2 = y[2];
  double d2F_ddelta2_prime = y[3];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  dfde[0] = dF_ddelta_prime;
  dfde[1] = 1.5*Omega_m/a*dF_ddelta - hubble*dF_ddelta_prime;
  dfde[2] = d2F_ddelta2_prime;
  dfde[3] = 1.5*Omega_m/a*(d2F_ddelta2 + 2.0*pow(dF_ddelta, 2)) + 6.0/2.0*pow(dF_ddelta_prime, 2) - hubble*d2F_ddelta2_prime;
  return GSL_SUCCESS;
}


int dF_cylindrical_ddelta_at_average_density_jac(double e, const double y[], double *dfdy, double dfde[], void *params){

  integration_parameters *integration_params = (integration_parameters *) params;
  Universe* pointer_to_Universe = integration_params->pointer_to_Universe;

  double dF_ddelta = y[0];
  double dF_ddelta_prime = y[1];
  double d2F_ddelta2 = y[2];
  double d2F_ddelta2_prime = y[3];
  double a = pointer_to_Universe->a_at_eta(e);
  double hubble = pointer_to_Universe->H_at_eta(e);
  double Omega_m = integration_params->Omega_m;

  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 4, 4);
  gsl_matrix * m = &dfdy_mat.matrix; 
  
  
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 1.0);
  gsl_matrix_set (m, 0, 2, 0.0);
  gsl_matrix_set (m, 0, 3, 0.0);

  gsl_matrix_set (m, 1, 0, 1.5*Omega_m/a);
  gsl_matrix_set (m, 1, 1, -hubble);
  gsl_matrix_set (m, 1, 2, 0.0);
  gsl_matrix_set (m, 1, 3, 0.0);
   
  gsl_matrix_set (m, 2, 0, 0.0);
  gsl_matrix_set (m, 2, 1, 0.0);
  gsl_matrix_set (m, 2, 2, 0.0);
  gsl_matrix_set (m, 2, 3, 1.0);

  gsl_matrix_set (m, 3, 0, 6.0*Omega_m/a*dF_ddelta);
  gsl_matrix_set (m, 3, 1, 12.0/2.0*dF_ddelta_prime);
  gsl_matrix_set (m, 3, 2, 1.5*Omega_m/a);
  gsl_matrix_set (m, 3, 3, -hubble);
  
  return GSL_SUCCESS;
}


