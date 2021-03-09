

struct integration_parameters_2D{

  double nonlinear_variance;
  double delta;

  complex<double> step;
  complex<double> dy_complex;
  
  vector<double> *coefficients_G;
  vector<double> *coefficients_G_prime;
  
};





int contour_integral_for_PDF_gsl(double y_real, const double y[], double dfde[], void *params){

  integration_parameters_2D *integration_params = (integration_parameters_2D *) params;   
  double dy_real = y_real-y[0];
  double dy_imag = dy_real/integration_params->dy_complex.real()*integration_params->dy_complex.imag();
  
  complex<double> z = complex<double>(y_real, y[1]+dy_imag);
  complex<double> tau = get_tau_from_secant_method_complex_Bernardeau_notation_2D(z, 0.0, integration_params->coefficients_G_prime);
  complex<double> G = return_polnomial_value(tau, integration_params->coefficients_G);
  complex<double> dy_complex;
  complex<double> exponent = (z*(G-integration_params->delta)-pow(tau, 2)*0.5)/integration_params->nonlinear_variance;
  dy_complex = conj(integration_params->delta-G)/abs(integration_params->delta-G);
    
  complex<double> step = dy_complex*exp(exponent);

  dfde[0] = 1.0;
  dfde[1] = dy_complex.imag()/dy_complex.real();
  dfde[2] = step.imag()/dy_complex.real();
  (*integration_params).step = step;
  (*integration_params).dy_complex = dy_complex;
  return GSL_SUCCESS;
}



int contour_integral_for_PDF_gsl_jac(double e, const double y[], double *dfdy, double dfde[], void *params){
  return GSL_SUCCESS;
}
