
/**********************************************
 **********************************************
 **_________ 5. OUTPUT AND CHECKS ___________**
 **********************************************
 ********************************************** 
 *                                            *
 * ..... 5.1 print_Newtonian_growth_factor    *
 * ..... 5.2 return_wave_numbers              *
 * ..... 5.3 transfer_function_at             *
 *                                            *
 **********************************************
 **********************************************/



/*******************************************************************************************************************************************************
 * 5.1 print_Newtonian_growth_factor
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

void Matter::print_Newtonian_growth_factor(string file_name){
  
  double e;
  double scale;
  double hubble;
  double z;
  double D;
  double D_prime;
  double d_eta, dD;
  
  fstream output;
  
  remove(file_name.c_str());
  FILE* F = fopen(file_name.c_str(),"w");
  fclose(F);
  output.open(file_name.c_str());
  
  
  output << setw(20) << "eta" << setw(20) << "a" << setw(20) << "H_conf" << setw(20) << "z" << setw(20) << "D(z)" << setw(20) << "D'(z)" << setw(20) << "D_phi(z)" << setw(20) << "D_phi'(z)" << '\n';
  
  for(int i = 1; i < this->number_of_entries_Newton; i++){
    e = this->eta_Newton[i];
    d_eta = e - this->eta_Newton[i-1];
    scale = this->universe->a_at_eta(e);
    hubble = this->universe->H_at_eta(e);
    z = 1.0/scale - 1.0;
    D = this->Newtonian_growth_factor_of_delta[i];
    D_prime = this->Newtonian_growth_factor_of_delta_prime[i];
    dD = D - this->Newtonian_growth_factor_of_delta[i-1];
    output << setw(20) << e << setw(20) << scale << setw(20) << hubble << setw(20) << z << setw(20) << D << setw(20) << D_prime << setw(20) << D/scale << setw(20) << D_prime/scale - D/scale*hubble << '\n';
  }
  
}

/*******************************************************************************************************************************************************
 * 5.2 return_wave_numbers 
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

vector<double> Matter::return_wave_numbers(){
  return this->wave_numbers;
}



/*******************************************************************************************************************************************************
 * 5.3 transfer_function_at 
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

double Matter::transfer_function_at(double k){
  return interpolate_neville_aitken(log(k), &this->log_wave_numbers, &this->transfer_function, this->order);
}



/*******************************************************************************************************************************************************
 * 5.9 compute_lognormal_final
 * (Version for 2D PDF only) 
 * Description:
 *  - Note: this is in the notation of Bernardeau rather than that of Valageas!
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/


void Matter::compute_lognormal_PDF_only(double R1, double eta, vector<double> *G_coeffs){
  
  double w = this->universe->eta_at_a(1.0) - eta;
  double log_R1 = log(R1);
  double sigma_R1_squared;
  double dsigma_R1_squared_dlogR;
  double D_growth = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double sL, dsL, sNL, cL, dcL1, dcL2, cNL;
  
  
  
  this->prepare_power_spectra_in_trough_format(w);
  
  
  double tau_prime_at_0;
  double tau_prime_prime_at_0;
  double F_prime_at_0 = D_growth;
  double F_prime_prime_at_0;
  double d2y_ddelta2_at_0;
  double ddelta_dy_at_0;
  double d2delta_dy2_at_0;
  double d2F_dy2;
  
  sL = variance_of_matter_within_R_2D();
  sNL = variance_of_matter_within_R_2D_NL();
  this->set_linear_Legendres(4, R1/w/c_over_e5);
  this->set_linear_Legendres(3, R1/w/c_over_e5);
  dsL = this->dcov1_at()*R1/w/c_over_e5*F_prime_at_0;
  
  //sigma_R1_squared = interpolate_neville_aitken(log_R1, &this->log_top_hat_radii, &this->top_hat_cylinder_variances, this->order);
  //dsigma_R1_squared_dlogR = interpolate_neville_aitken_derivative(log_R1, &this->log_top_hat_radii, &this->top_hat_cylinder_variances, this->order);  
  sigma_R1_squared = sL/D_growth/D_growth;
  dsigma_R1_squared_dlogR = 2.0*dsL/pow(D_growth, 3);
  tau_prime_at_0 = 1.0/sqrt(sigma_R1_squared);
  tau_prime_prime_at_0 = -0.5*F_prime_at_0/pow(sigma_R1_squared, 1.5)*dsigma_R1_squared_dlogR;
  F_prime_prime_at_0 = interpolate_neville_aitken(eta, &this->eta_NL, &this->F_prime_prime_of_eta, this->order);
  ddelta_dy_at_0 = sigma_R1_squared*F_prime_at_0;
  d2y_ddelta2_at_0 = tau_prime_prime_at_0*tau_prime_at_0/F_prime_prime_at_0 + 2.0*(tau_prime_at_0*tau_prime_prime_at_0/F_prime_at_0 - pow(tau_prime_at_0/F_prime_at_0, 2)*F_prime_prime_at_0);
  d2delta_dy2_at_0 = -pow(ddelta_dy_at_0, 3)*d2y_ddelta2_at_0;
  
  d2F_dy2 = F_prime_prime_at_0*pow(ddelta_dy_at_0, 2) - F_prime_at_0*d2y_ddelta2_at_0*pow(ddelta_dy_at_0, 3);
  
  double A, B;
  
  
  (*G_coeffs)[2] = sL/2.0;
  (*G_coeffs)[3] = d2F_dy2/6.0;
    
  for(int i = 0; i < 4; i++){
    if(i > 1){
      A = pow(sL, i-1);
      B = pow(sNL, i-1);
      (*G_coeffs)[i] *= B/A;
    }
    else{
      (*G_coeffs)[i] = 0.0;
    }
  }
    
  // set linear coefficient of generating to covariance (is it should be).
  (*G_coeffs)[0] = 0.0;
  (*G_coeffs)[1] = 0.0;   
  (*G_coeffs)[2] = sNL/2.0;

}





/*******************************************************************************************************************************************************
 * 5.9 compute_lognormal_final
 * (Version for 2D PDF only) 
 * Description:
 *  - Note: this is in the notation of Bernardeau rather than that of Valageas!
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/


void Matter::compute_lognormal_old(double R1, double eta, vector<double> bins, vector<double> *G_coeffs, vector<vector<double> > *G_kappa_coeffs){
  
  double w = this->universe->eta_at_a(1.0) - eta;
  double log_R1 = log(R1);
  double sigma_R1_squared;
  double dsigma_R1_squared_dlogR;
  double D_growth = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double vL, dvL, dsL, vNL, cL, dcL1, dcL2, cNL;
  
  
  
  this->prepare_power_spectra_in_trough_format(w);
  
  
  double tau_prime_at_0;
  double tau_prime_prime_at_0;
  double F_prime_at_0 = D_growth;
  double F_prime_prime_at_0;
  double d2y_ddelta2_at_0;
  double ddelta_dy_at_0;
  double dy_ddelta_at_0;
  double d2delta_dy2_at_0;
  double dF_dy;
  double d2F_dy2;
  
  vL = variance_of_matter_within_R_2D();
  vNL = variance_of_matter_within_R_2D_NL();
  this->set_linear_Legendres(4, R1/w/c_over_e5);
  this->set_linear_Legendres(3, R1/w/c_over_e5);
  dvL = 2.0*this->dcov1_at()*R1/w/c_over_e5;
  
  
  /*
   * prime here means derivative wrt. delta_L (see Valageas 2002)
   * 
   * dphi/dy      = F(delta_L(y))
   * F(0)         = 0
   * tau(delta_L) = delta_L/sigma_(sqrt[1+F]R)
   * tau'(0)      = 1/sigma_(R)
   * tau''(0)     = -0.5/sigma^1.5_(R)*dsigma^2_(R)/dR * d[sqrt[1+F]R]/ddelta
   *              = -0.5/sigma^1.5_(R)*dsigma^2_(R)/dlnR * d[sqrt[1+F]]/ddelta
   *              = -0.25/sigma^1.5_(R)*dsigma^2_(R)/dlnR * F'(0)
   * 
   */ 
    
  sigma_R1_squared = vL/D_growth/D_growth;
  dsigma_R1_squared_dlogR = dvL/D_growth/D_growth;
  tau_prime_at_0 = 1.0/sqrt(sigma_R1_squared);
  tau_prime_prime_at_0 = -2.0*0.25*F_prime_at_0/pow(sigma_R1_squared, 1.5)*dsigma_R1_squared_dlogR;
  F_prime_prime_at_0 = interpolate_neville_aitken(eta, &this->eta_NL, &this->F_prime_prime_of_eta, this->order);

  ddelta_dy_at_0 = sigma_R1_squared*F_prime_at_0;
  dy_ddelta_at_0 = tau_prime_at_0*tau_prime_at_0/F_prime_at_0;
  dF_dy = F_prime_at_0*ddelta_dy_at_0;
    
  // The following formulae are identical:
  //d2y_ddelta2_at_0 = 3.0*tau_prime_prime_at_0*tau_prime_at_0/F_prime_at_0 - 2.0*tau_prime_at_0*tau_prime_at_0*F_prime_prime_at_0/F_prime_at_0/F_prime_at_0;
  d2y_ddelta2_at_0 = 2.0*(-F_prime_prime_at_0/F_prime_at_0/F_prime_at_0/sigma_R1_squared - 3.0/(4.0*sigma_R1_squared*sigma_R1_squared)*dsigma_R1_squared_dlogR);
  
  
  d2delta_dy2_at_0 = -pow(ddelta_dy_at_0, 3)*d2y_ddelta2_at_0;
  d2F_dy2 = F_prime_prime_at_0*pow(ddelta_dy_at_0, 2) + F_prime_at_0*d2delta_dy2_at_0;
  
  
  double D_11 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double D_22 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_second_order, this->order);
  double mu = 1.0 - D_22/D_11/D_11;
  
  /*
  cout << '\n';
  cout << setw(20);
  cout << 34.0/7.0 + 1.5*dvL/vL;
  cout << setw(20);
  cout << 36.0/7.0 + 1.5*dvL/vL;
  cout << setw(20);
  cout << 3.0*(1.0+mu) + 1.5*dvL/vL;
  cout << setw(20);
  cout << d2F_dy2/vL/vL;
  cout << '\n';
  */

  double A, B;
  double auxilliary_1_over_w_c = 1.0/w/c_over_e5;
  double auxilliary_1_over_w_c_F_prime = 1.0/w/c_over_e5*F_prime_at_0;
    
  // make it derivative wrt. delta_L:
  dvL *= 0.5*F_prime_at_0;
  
  for(int b = 0; b < bins.size(); b++){
    
    this->set_linear_Legendres(3, bins[b]*auxilliary_1_over_w_c);
    cL = covariance_of_matter_within_R_2D();
    dcL1 = 0.5*this->dcov1_at()*R1*auxilliary_1_over_w_c_F_prime;
    dcL2 = 0.5*this->dcov2_at()*bins[b]*auxilliary_1_over_w_c_F_prime;
    cNL = covariance_of_matter_within_R_2D_NL();
    
    (*G_kappa_coeffs)[b][0] = 0.0;
    (*G_kappa_coeffs)[b][1] = cL;
    (*G_kappa_coeffs)[b][2] = 2.0*vL*(dcL1 + cL/vL*(dcL2 - dvL))/pow(D_growth, 2) + d2delta_dy2_at_0*cL/vL;
    (*G_kappa_coeffs)[b][2] = 0.5*(D_growth*(*G_kappa_coeffs)[b][2] + F_prime_prime_at_0*pow(cL/D_growth, 2));
    
    
    for(int i = 0; i < 3; i++){
      if(i > 1){
        A = pow(cL, i) + double(i)*cL*pow(vL, i-1);
        B = pow(cNL, i) + double(i)*cNL*pow(vNL, i-1);
        (*G_kappa_coeffs)[b][i] *= B/A;
      }
      if(i == 1){
        (*G_kappa_coeffs)[b][i] = cNL;
      }
      if(i == 0){
        (*G_kappa_coeffs)[b][i] = 0.0;
      }
    }
    
  }
    
  (*G_coeffs)[2] = vL/2.0;
  (*G_coeffs)[3] = d2F_dy2/6.0;
    
  for(int i = 0; i < 4; i++){
    if(i > 1){
      A = pow(vL, i-1);
      B = pow(vNL, i-1);
      (*G_coeffs)[i] *= B/A;
    }
    else{
      (*G_coeffs)[i] = 0.0;
    }
  }
    
  // set linear coefficient of generating to covariance (is it should be).
  (*G_coeffs)[0] = 0.0;
  (*G_coeffs)[1] = 0.0;   
  (*G_coeffs)[2] = vNL/2.0;

}

void Matter::compute_lognormal_final(double R1, double eta, vector<double> *bins, vector<double> *G_coeffs, vector<vector<double> > *G_kappa_coeffs){
  
  double w = this->universe->eta_at_a(1.0) - eta;
  double log_R1 = log(R1);
  double sigma_R1_squared;
  double dsigma_R1_squared_dlogR;
  double D_growth = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double vL, dvL_dlnR, dlnvL_dlnR, dsL, vNL, cL, dcL1, dlncL1, dcL2, dlncL2, cNL;
  
  this->prepare_power_spectra_in_trough_format(w);
  
  double tau_prime_at_0;
  double tau_prime_prime_at_0;
  double F_prime_at_0 = D_growth;
  double F_prime_prime_at_0;
  double d2y_ddelta2_at_0;
  double ddelta_dy_at_0;
  double d2delta_dy2_at_0;
  double d2F_dy2;
  
  vL = variance_of_matter_within_R_2D();
  vNL = variance_of_matter_within_R_2D_NL();
  //cout << vL << "   HAHAHAHA\n";
  //exit(1);
  dvL_dlnR = 0.0;
  for(int ell = 0; ell < this->ell_max; ell++){
    dvL_dlnR += this->current_P_delta_L_in_trough_format[ell]*this->theta_trough_Legendres[ell]*this->dtheta_trough_Legendres_dtheta_trough[ell];
  }
  /*
    TOM WAS HERE
    - To get rid of divisions /c_over_e5 is replaced with *inverse_c_over_e5
   */
  dvL_dlnR *= 2.0*R1/w*inverse_c_over_e5;
  //dvL_dlnR *= 2.0*R1/w/c_over_e5;
  dlnvL_dlnR = dvL_dlnR/vL;
  
  
  double D_11 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double D_22 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_second_order, this->order);
  double mu = 1.0 - D_22/D_11/D_11;
  double one_plus_mu = (1.0+mu);
  /*
    TOM WAS HERE
    - To get rid of divisions /c_over_e5 is replaced with *inverse_c_over_e5
    - moved the 0.5 from in the loop to onto these variables
   */
  double auxilliary_1_over_w_c = 0.5/w*inverse_c_over_e5;
  double auxilliary_1_over_w_c_F_prime = 0.5/w*inverse_c_over_e5*F_prime_at_0;
  //double auxilliary_1_over_w_c = 1.0/w/c_over_e5;
  //double auxilliary_1_over_w_c_F_prime = 1.0/w/c_over_e5*F_prime_at_0;
  
  /*
    TOM WAS HERE
    - if a function is used as a conditional in a loop then that function will be called at every iteration.
    In this, we can save the bin size ahead of time and get rid of calls to the size() function.
   */
  int binsize = bins->size();
  for(int b = 0; b < binsize; b++){
    //for(int b = 0; b < bin->size(); b++){
    
    cL = 0.0;
    dcL1 = 0.0;
    dcL2 = 0.0;
    cNL = 0.0;
    
    for(int ell = 0; ell < this->ell_max; ell++){
      cL += this->current_P_delta_L_in_trough_format_times_trough_legendres[ell]*this->bin_Legendres[b][ell];
      cNL += this->current_P_delta_NL_in_trough_format_times_trough_legendres[ell]*this->bin_Legendres[b][ell];
      dcL1 += this->current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough[ell]*this->bin_Legendres[b][ell];
      dcL2 += this->current_P_delta_L_in_trough_format_times_trough_legendres[ell]*this->dbin_Legendres_dtheta_bin[b][ell];
    }
    
    /*
      TOM WAS HERE
      - moved the 0.5 from in here (in the loop) to up where auxilliary_1_over_w_c is defined
    */
    dcL1 *= R1*auxilliary_1_over_w_c_F_prime;
    dcL2 *= (*bins)[b]*auxilliary_1_over_w_c_F_prime;
    //dcL1 *= 0.5*R1*auxilliary_1_over_w_c_F_prime;
    //dcL2 *= 0.5*(*bins)[b]*auxilliary_1_over_w_c_F_prime;
    
    dlncL1 = dcL1/cL;
    dlncL2 = dcL2/cL;
    /*
    cout << w << setw(20);
    cout << (*bins)[b] << setw(20);
    cout << R1 << setw(20);
    cout << vNL*cNL*(one_plus_mu*2.0 + 0.5*dlnvL_dlnR + dlncL1) << setw(20);
    cout << cNL*cNL*(one_plus_mu + dlncL2) << setw(20);
    cout << one_plus_mu*(2.0*vNL*cNL + cNL*cNL) << setw(20);
    cout << cNL*cNL*dlncL2 + 0.5*cNL*vNL*dlnvL_dlnR + vNL*cNL*dlncL1 << setw(20);
    cout << '\n';*/
//cin >> cL;
    
    (*G_kappa_coeffs)[b][0] = 0.0;
    (*G_kappa_coeffs)[b][1] = cNL;
    (*G_kappa_coeffs)[b][2] = one_plus_mu*(2.0*vNL*cNL + cNL*cNL) + cNL*cNL*dlncL2 + 0.5*cNL*vNL*dlnvL_dlnR + vNL*cNL*dlncL1;
    (*G_kappa_coeffs)[b][2] *= 0.5*(1.0+this->S3_enhance);
    
  }

  /*
    TOM WAS HERE
    - divisions are ~20 times slower than mutliplications. Getting rid of as many as possible can help.
    - In this case, the divisions in the next four lines happen rarely enough that this was negligible.
   */
  (*G_coeffs)[0] = 0.0;
  (*G_coeffs)[1] = 0.0;
  (*G_coeffs)[2] = vNL*0.5;
  (*G_coeffs)[3] = vNL*vNL*(0.5*one_plus_mu + 0.25*dlnvL_dlnR)*(1.0+this->S3_enhance);
  //(*G_coeffs)[2] = vNL/2.0;
  //(*G_coeffs)[3] = vNL*vNL*(3.0*one_plus_mu + 1.5*dlnvL_dlnR)/6.0;
  
  //cout << w << setw(20) << 36.0/7.0 << setw(20) << 3.0*one_plus_mu << setw(20) << 1.5*dlnvL_dlnR << '\n';

}

//Marco was here
void Matter::compute_variance_and_skewness_arr(int size_theta, vector<double> R1, double eta, vector<vector<double>> *G_coeffs){

  double w = this->universe->eta_at_a(1.0) - eta;
  vector<double> dummy_array(size_theta, 0.0);
  double dummy = 0.0;
  int dummy_index = 0;
    int mute = 0;
  vector<double> log_R1(size_theta,0.0);
  double a,b,c;
  double a1,a2,a3,a4,a5,a6,a7,a8,a9;

  vector<double> dvL_dlnR(size_theta,0.0);
  vector<double> dvL_dlnR_b(size_theta,0.0);
  vector<double> dlnvL_dlnR (size_theta,0.0);
  vector<double> dlnvL_dlnR_b (size_theta,0.0);
  vector<double> vNL(size_theta,0.0);
  vector<double> vNL_notmasked(size_theta,0.0);
  vector<double> vL(size_theta,0.0);
  vector<double> vL_notmasked(size_theta,0.0);

  vector<double> vNL_a(size_theta,0.0);
  vector<double> vL_a(size_theta,0.0);
  vector<double> vNL_b(size_theta,0.0);
  vector<double> vL_b(size_theta,0.0);
  vector<double> vNL_c(size_theta,0.0);
  vector<double> vL_c(size_theta,0.0);


  if (this->NL_p == 1){
      //SC01
      a1 = 0.25;
      a2 = 3.5;
      a3 = 2;
      a4 = 1;
      a5 = 2;
      a6 = -0.2;
      a7 = 1;
      a8 = 0;
      a9 = 0; 
  }
  if (this->NL_p== 2){
      //GM12
      a1 = 0.484;
      a2 = 3.740;
      a3 = -0.849;
      a4 = 0.392;
      a5 = 1.013;
      a6 = -0.575;
      a7 = 0.128;
      a8 = -0.722;
      a9 = -0.926;
  }



  double ns=0;
  double s8=0;
  
  ns = this->current_n_eff;
  
  s8 =  this->cosmology.sigma_8;
  double Dp = this->Dp;



  double sigma_R1_squared;
  double dsigma_R1_squared_dlogR;

  double D_growth = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double  dsL, cL, dcL1, dlncL1, dcL2, dlncL2, cNL;

  double d1,ln_k1 ,ln_k1_1;

  this->prepare_power_spectra_in_trough_format_2(w);


  double tau_prime_at_0;
  double tau_prime_prime_at_0;
  double F_prime_at_0 = D_growth;
  double F_prime_prime_at_0;
  double d2y_ddelta2_at_0;
  double ddelta_dy_at_0;
  double d2delta_dy2_at_0;
  double d2F_dy2;


  /*
  
  I don't remember how I was doing it.
  Technically, I shouldn't mask vNl_a,b,c here so I'll be doing a hack.
  I probably hacked the emulator a while a go, producing runs
  with and without masks. Probably.
  
  */

  if(this->len_mask > 0){
      // compute them masked.
      mute = this->len_mask;
      variance_of_matter_within_R_2D_arr(size_theta,&vL);
      variance_of_matter_within_R_2D_NL_arr(size_theta,&vNL);
      this->len_mask = 0;
      
      variance_of_matter_within_R_2D_arr(size_theta,&vL_notmasked);
      variance_of_matter_within_R_2D_NL_arr(size_theta,&vNL_notmasked);
      variance_of_matter_within_R_2D_NL_arr_mod(size_theta,&vNL_a,&vNL_b,&vNL_c,w); // routine in matter_variance
      variance_of_matter_within_R_2D_arr_mod(size_theta,&vL_a,&vL_b,&vL_c,w); // routine in matter_variance

      this->len_mask = mute;
  }else{
  variance_of_matter_within_R_2D_arr(size_theta,&vL_notmasked);
  variance_of_matter_within_R_2D_NL_arr(size_theta,&vNL_notmasked);
  variance_of_matter_within_R_2D_NL_arr_mod(size_theta,&vNL_a,&vNL_b,&vNL_c,w); // routine in matter_variance
  variance_of_matter_within_R_2D_arr_mod(size_theta,&vL_a,&vL_b,&vL_c,w); // routine in matter_variance
  variance_of_matter_within_R_2D_arr(size_theta,&vL);
  variance_of_matter_within_R_2D_NL_arr(size_theta,&vNL); 
      
  }
    
    


    

  double q = 0;

  // no mask, no pixel window ************************
  if (this->len_mask == 0){
    if (this->len_pix_mask == 0){
    for(int ell = 1; ell < ell_max; ell++){
        for(int i=0; i<size_theta; i++){
            dvL_dlnR[i] += this->current_P_delta_L_in_trough_format[ell]*this->Atheta_trough_Legendres[i][ell]*this->Adtheta_trough_Legendres_dtheta_trough[i][ell];
        }
    }
    }else{

    // no mask, yes pix window ***************************
    double b =0;
    for(int ell = 1; ell <  this->len_pix_mask; ell++){
        q= ell/l_nl;

        if (this->NL_p == 0){
            a = 1.;
            b = 1.;
            c = 1.;
        }else{
            ln_k1 = log((ell+0.5)/(w)-(ell+0.5)/(w)/20.);
            ln_k1_1 = log((ell+0.5)/(w)+(ell+0.5)/(w)/20.);
            d1 = log((ell+0.5)/(w*4344.*0.7)*21./20.) - log((ell+0.5)/(w*4344.*0.7)*19./20.);
            ns = (log(current_P_L_at(ln_k1_1))-log(current_P_L_at(ln_k1)))/d1;
            a = (1. + pow(s8*Dp,a6)*pow((0.7*(4-pow(2,ns))/(1+pow(2,2*ns+1))),0.5)*pow(q*a1,ns+a2))/(1+pow(q*a1,ns+a2));
            b = (1. + 0.2*a3*(ns+3)*pow(q*a7,ns+a8+3))/(1+pow(q*a7,ns+a8+3.5));
            c = (1. + 4.5*a4/(1.5+pow(ns+3,4))*pow(q*a5,ns+3+a9))/(1+pow(q*a5,ns+3.5+a9));
        }
        
        for(int i=0; i<size_theta; i++){

            dvL_dlnR[i] += this->current_P_delta_L_in_trough_format[ell]*this->Atheta_trough_Legendres[i][ell]*this->Adtheta_trough_Legendres_dtheta_trough[i][ell]*Cl_mask[ell]*Cl_mask[ell];
            dvL_dlnR_b[i] += b*this->current_P_delta_L_in_trough_format[ell]*this->Atheta_trough_Legendres[i][ell]*this->Adtheta_trough_Legendres_dtheta_trough[i][ell]*Cl_mask[ell]*Cl_mask[ell];

    }
    }
    }
    }else{
       //cout <<"solo qui"<<endl;
       for(int ell = 1; ell <  this->len_pix_mask; ell++){
        q= ell/l_nl;
        ln_k1 = log((ell+0.5)/(w)-(ell+0.5)/(w)/20.);
        ln_k1_1 = log((ell+0.5)/(w)+(ell+0.5)/(w)/20.);
        d1 = log((ell+0.5)/(w*4344.*0.7)*21./20.) - log((ell+0.5)/(w*4344.*0.7)*19./20.);
        ns = (log(current_P_L_at(ln_k1_1))-log(current_P_L_at(ln_k1)))/d1;
        if (this->NL_p == 0){
            a = 1.;
            b = 1.;
            c = 1.;
        }else{
            a = (1. + pow(s8*Dp,a6)*pow((0.7*(4-pow(2,ns))/(1+pow(2,2*ns+1))),0.5)*pow(q*a1,ns+a2))/(1+pow(q*a1,ns+a2));
            b = (1. + 0.2*a3*(ns+3)*pow(q*a7,ns+a8+3))/(1+pow(q*a7,ns+a8+3.5));
            c = (1. + 4.5*a4/(1.5+pow(ns+3,4))*pow(q*a5,ns+3+a9))/(1+pow(q*a5,ns+3.5+a9));
        }
    
           
        for(int i=0; i<size_theta; i++){
            dvL_dlnR[i] += this->current_P_delta_L_in_trough_format[ell]*this->Atheta_trough_Legendres[i][ell]*this->Adtheta_trough_Legendres_dtheta_trough[i][ell]*Cl_mask[ell]*Cl_mask[ell];
            dvL_dlnR_b[i] += b*this->current_P_delta_L_in_trough_format[ell]*this->Atheta_trough_Legendres[i][ell]*this->Adtheta_trough_Legendres_dtheta_trough[i][ell]*Cl_mask[ell]*Cl_mask[ell];

}
}
}


  for (int i = 0; i< size_theta; i++){
    log_R1[i] = log(R1[i]);
     dvL_dlnR[i] *= 2.0*R1[i]/w*inverse_c_over_e5;
     dvL_dlnR_b[i] *= 2.0*R1[i]/w*inverse_c_over_e5;
     dlnvL_dlnR[i] = dvL_dlnR[i]/vL_notmasked[i];
     dlnvL_dlnR_b[i] = dvL_dlnR_b[i]/vL_b[i];
  }


  /*
    TOM WAS HERE
    - To get rid of divisions /c_over_e5 is replaced with *inverse_c_over_e5
   */

  //dvL_dlnR *= 2.0*R1/w/c_over_e5;



  double D_11 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double D_22 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_second_order, this->order);
  double mu = 1.0 - D_22/D_11/D_11;
  double one_plus_mu = (1.0+mu);

  /*
    TOM WAS HERE
    - To get rid of divisions /c_over_e5 is replaced with *inverse_c_over_e5
    - moved the 0.5 from in the loop to onto these variables
   */
  double auxilliary_1_over_w_c = 0.5/w*inverse_c_over_e5;
  double auxilliary_1_over_w_c_F_prime = 0.5/w*inverse_c_over_e5*F_prime_at_0;


  /*
    TOM WAS HERE
    - divisions are ~20 times slower than mutliplications. Getting rid of as many as possible can help.
    - In this case, the divisions in the next four lines happen rarely enough that this was negligible.
   */

  //cout << vNL_a[9]<<' '<< vNL_b[9]<<' ' << vNL_c[9]<<' ' <<w <<endl;
  //cout <<mu<<endl;
  //cout << dlnvL_dlnR_b[9]<<endl;
  //cout << 'boia' <<endl;
  //cout << vNL[9]*vNL[9]*(0.5*one_plus_mu + 0.25*dlnvL_dlnR[9])*(1.0+this->S3_enhance)<<endl;
  //cout << vNL[9]*vNL[9]<<endl;
  //cout << (1.0+this->S3_enhance)<<endl;
  //cout <<'_'<<endl;
      
  for (int i = 0; i< size_theta; i++){
    //cout<<i<<"  "<<vNL[i]<<"   "<<vNL_a[i]<<"   "<<vNL_b[i]<<"   "<<vNL_a[i]<<endl;
    //cout<<i<<"  "<<vNL[i]*vNL[i]*one_plus_mu<<"   "<<(2*vNL_a[i]*vNL_a[i] -(1.-mu)*vNL_c[i]*vNL_c[i])<<endl;
    
      
     // cout<<i<<"  "<<dlnvL_dlnR[i]<< "   "<<dlnvL_dlnR_b[i]<<endl;
    (*G_coeffs)[i][0] = 0.0;
    (*G_coeffs)[i][1] = 0.0;
    (*G_coeffs)[i][2] = vNL[i]*0.5;
    (*G_coeffs)[i][3] = vNL[i]*vNL[i]*(0.5*one_plus_mu + 0.25*dlnvL_dlnR[i])*(1.0+this->S3_enhance);
    //mu = 5./7.;
      
    // S3 is not affected by mask. first term goes as M^2 * f_sky (since M already goes as 1/f_sky)
    (*G_coeffs)[i][3] = (1./(this->fact_area))*vNL[i]*vNL[i] * (0.5*(2*vNL_a[i]*vNL_a[i] -(1.-mu)*vNL_c[i]*vNL_c[i])+ vNL_b[i]*vNL_b[i]*0.25*dlnvL_dlnR_b[i])/vNL_notmasked[i]/vNL_notmasked[i];

      // old( wrong?)
    //(*G_coeffs)[i][3] = (0.5*(2*vNL_a[i]*vNL_a[i] -(1.-mu)*vNL_c[i]*vNL_c[i])+ vNL_b[i]*vNL_b[i]*0.25*dlnvL_dlnR_b[i]);

    //cout<<inverse_c_over_e5<<endl;
   }

  //(*G_coeffs)[2] = vNL/2.0;
  //(*G_coeffs)[3] = vNL*vNL*(3.0*one_plus_mu + 1.5*dlnvL_dlnR)/6.0;

}



void Matter::compute_variance_and_skewness(double R1, double eta, vector<double> *G_coeffs){
  
  double w = this->universe->eta_at_a(1.0) - eta;
  double log_R1 = log(R1);
  double sigma_R1_squared;
  double dsigma_R1_squared_dlogR;

  double D_growth = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double vL, dvL_dlnR, dlnvL_dlnR, dsL, vNL, cL, dcL1, dlncL1, dcL2, dlncL2, cNL;

  this->prepare_power_spectra_in_trough_format(w);

  double tau_prime_at_0;
  double tau_prime_prime_at_0;
  double F_prime_at_0 = D_growth;
  double F_prime_prime_at_0;
  double d2y_ddelta2_at_0;
  double ddelta_dy_at_0;
  double d2delta_dy2_at_0;
  double d2F_dy2;
  
  vL = variance_of_matter_within_R_2D();
  vNL = variance_of_matter_within_R_2D_NL(); // routin in matter_variance
  //cout << vL << "   HAHAHAHA\n";
  //exit(1);
  dvL_dlnR = 0.0;
  for(int ell = 0; ell < this->ell_max; ell++){
    dvL_dlnR += this->current_P_delta_L_in_trough_format[ell]*this->theta_trough_Legendres[ell]*this->dtheta_trough_Legendres_dtheta_trough[ell];
  }
  /*
    TOM WAS HERE
    - To get rid of divisions /c_over_e5 is replaced with *inverse_c_over_e5
   */
  dvL_dlnR *= 2.0*R1/w*inverse_c_over_e5;
  //dvL_dlnR *= 2.0*R1/w/c_over_e5;
  dlnvL_dlnR = dvL_dlnR/vL;
  
  
  double D_11 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double D_22 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_second_order, this->order);
  double mu = 1.0 - D_22/D_11/D_11;
  double one_plus_mu = (1.0+mu);
  
  /*
    TOM WAS HERE
    - To get rid of divisions /c_over_e5 is replaced with *inverse_c_over_e5
    - moved the 0.5 from in the loop to onto these variables
   */
  double auxilliary_1_over_w_c = 0.5/w*inverse_c_over_e5;
  double auxilliary_1_over_w_c_F_prime = 0.5/w*inverse_c_over_e5*F_prime_at_0;


  /*
    TOM WAS HERE
    - divisions are ~20 times slower than mutliplications. Getting rid of as many as possible can help.
    - In this case, the divisions in the next four lines happen rarely enough that this was negligible.
   */
  (*G_coeffs)[0] = 0.0;
  (*G_coeffs)[1] = 0.0;
  (*G_coeffs)[2] = vNL*0.5;
  (*G_coeffs)[3] = vNL*vNL*(0.5*one_plus_mu + 0.25*dlnvL_dlnR)*(1.0+this->S3_enhance);
  //(*G_coeffs)[2] = vNL/2.0;
  //(*G_coeffs)[3] = vNL*vNL*(3.0*one_plus_mu + 1.5*dlnvL_dlnR)/6.0;
  
}




void Matter::compute_Bernardeau_final(double R1, double eta, vector<double> bins, vector<double> *G_coeffs, vector<vector<double> > *G_kappa_coeffs, vector<double> *lambda_of_w, vector<double> *phi_lambda_of_w, vector<double> *phi_prime_lambda_of_w, vector<vector<double> > *G_kappa_of_w){

  double w = this->universe->eta_at_a(1.0) - eta;
  double log_R1 = log(R1);
  double sigma_R1_squared;
  double dsigma_R1_squared_dlogR;
  double D_growth = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double vL, dvL_dlnR, dlnvL_dlnR, dsL, vNL, cL, dcL1, dlncL1, dcL2, dlncL2, cNL;
  
  this->prepare_power_spectra_in_trough_format(w);
  
  double tau_prime_at_0;
  double tau_prime_prime_at_0;
  double F_prime_at_0 = D_growth;
  double F_prime_prime_at_0;
  double d2y_ddelta2_at_0;
  double ddelta_dy_at_0;
  double d2delta_dy2_at_0;
  
  vL = variance_of_matter_within_R_2D();
  vNL = variance_of_matter_within_R_2D_NL();
  dvL_dlnR = 0.0;
  for(int ell = 0; ell < this->ell_max; ell++){
    dvL_dlnR += this->current_P_delta_L_in_trough_format[ell]*this->theta_trough_Legendres[ell]*this->dtheta_trough_Legendres_dtheta_trough[ell];
  }

  dvL_dlnR *= 2.0*R1/w*inverse_c_over_e5;

  //dlnvL_dlnR = dlnvL_dlnRvL; // not declared in this scope!
  dlnvL_dlnR = dvL_dlnR; //I MADE THIS CHANGE BUT THIS NEED TO BE FIXED!
  
  double D_11 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  double D_22 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_second_order, this->order);
  double mu = 1.0 - D_22/D_11/D_11;
  double one_plus_mu = (1.0+mu)*(1.0+this->S3_enhance);
  
  /*
   * 
   * <---------------------
   * 
   */
  
  
  int n;
  int n_reduced;
  int i_min;
  int i_max;
  int n_aux = 30;
  int polynomial_order = constants::coeff_order;
  
  vector<double> delta_values(0, 0.0);
  vector<double> F_values(0, 0.0);
  vector<double> F_prime_values(0, 0.0);
  
  
  /*
   * From here, the generating function of delta_R is computed.
   * 
   * --------------------->
   * 
   */

  this->return_delta_NL_of_delta_L_and_dF_ddelta(eta, &delta_values, &F_values, &F_prime_values);
  n = delta_values.size();
  
    
  this->set_linear_Legendres(4, R1/w/c_over_e5);
  this->set_linear_Legendres(3, R1/w/c_over_e5);
  dvL_dlnR = this->dcov1_at()*R1/w/c_over_e5*F_prime_at_0;

  vector<double> tau_prime(n, 0.0);
  vector<double> tau_values(n, 0.0);
  vector<double> y_values(n, 0.0);
  vector<double> phi_values(n, 0.0);
  
  //vector<double> aux_coeffs_G_kappa2(polynomial_order+1, 0.0);
  //compute_phi_of_y_and_mean_delta2_cylinder(R1, bins[20], eta, &y_reduced, &phi_reduced, &aux_coeffs_G_kappa2);
  

  for(int i = 0; i < n; i++){
    sigma_R1_squared = interpolate_neville_aitken(log_R1 + 0.5*log(1.0+F_values[i]) - log(c_over_e5), &this->log_top_hat_radii, &this->top_hat_cylinder_variances, this->order);  
    dsigma_R1_squared_dlogR = interpolate_neville_aitken_derivative(log_R1 + 0.5*log(1.0+F_values[i]) - log(c_over_e5), &this->log_top_hat_radii, &this->top_hat_cylinder_variances, this->order);  
    tau_values[i] = delta_values[i]/sqrt(sigma_R1_squared);
/*
    set_permanent_Legendres(constants::ell_max, R1*sqrt(1.0+F_values[i])/w/c_over_e5);
    this->prepare_power_spectra_in_trough_format(w);
    double vL_dummy = variance_of_matter_within_R_2D();
    
    cout << R1*sqrt(1.0+F_values[i]) << setw(20);
    cout << D_growth*D_growth*sigma_R1_squared << setw(20);
    cout << vL << setw(20);
    cout << vL_dummy/(D_growth*D_growth*sigma_R1_squared) << setw(20);
    cout << D_growth*delta_values[i] << setw(20);
    cout << F_values[i] << '\n';*/
    
    /* Tripple checked:*/
    tau_prime[i] = (1.0 - 0.25*dsigma_R1_squared_dlogR*F_prime_values[i]/(1.0+F_values[i])*delta_values[i]/sigma_R1_squared)/sqrt(sigma_R1_squared);
    
    y_values[i] = tau_values[i]*tau_prime[i]/F_prime_values[i];
    phi_values[i] = y_values[i]*F_values[i] - 0.5*pow(tau_values[i], 2.0);
    
  }
  //exit(1);

  i_min = find_index(0.0, &tau_values)-3;
  i_max = find_index(0.0, &tau_values)+3;
  while(y_values[i_min-1] < y_values[i_min] && i_min > 0) i_min--;
  while(y_values[i_max+1] > y_values[i_max]  && i_max < n-1) i_max++;

  n_reduced = i_max - i_min + 1;
  vector<double> tau_values_reduced(n_reduced, 0.0);
  vector<double> y_values_reduced(n_reduced, 0.0);
  vector<double> delta_values_reduced(n_reduced, 0.0);
  vector<double> F_values_reduced(n_reduced, 0.0);

  
  (*lambda_of_w) = vector<double>(n_reduced, 0.0);
  (*phi_lambda_of_w) = vector<double>(n_reduced, 0.0);
  (*phi_prime_lambda_of_w) = vector<double>(n_reduced, 0.0);
  (*G_kappa_of_w) = vector<vector<double> >(bins.size(), vector<double>(n_reduced, 0.0));
  
  // This gives the rescaled gen. function phi(y) = Sum <d^n>_{c, NL} y^n.     
  for(int i = 0; i < n_reduced; i++){
    tau_values_reduced[i] = tau_values[i+i_min];
    y_values_reduced[i] = y_values[i+i_min];
    delta_values_reduced[i] = delta_values[i+i_min];
    F_values_reduced[i] = F_values[i+i_min];
    (*lambda_of_w)[i] = vL/vNL*y_values[i+i_min];
    (*phi_lambda_of_w)[i] = vL/vNL*phi_values[i+i_min];
    (*phi_prime_lambda_of_w)[i] = F_values[i+i_min];
  }
  
  
  /*
   * 
   * <---------------------
   * 
   */
  
  
  
  
  
  /*
   * From here, the y-coefficients of <kappa | delta(y)> are computed:
   * 
   * --------------------->
   * 
   */

  vector<double> tau_aux(n_aux, 0.0);
  vector<double> y_aux(n_aux, 0.0);
  vector<double> G(n_aux, 0.0);
  vector<double> G_kappa(n_aux, 0.0);
  vector<double> aux_coeffs_G(polynomial_order+1, 0.0);
  vector<double> aux_coeffs_G_kappa(polynomial_order+1, 0.0);
  vector<double> aux_coeffs_y(polynomial_order+1, 0.0);
  vector<double> fact = factoria(polynomial_order+1);
      
  double tau_max = tau_values[i_max];
  double tau_min = tau_values[i_min];
  double dtau = -tau_min/double(n_aux/2);
  for(int i = 0; i < n_aux/2; i++){
    tau_aux[i] = tau_min + double(i)*dtau;
  }
  tau_aux[n_aux/2] = 0.0;
  dtau = tau_max/double(n_aux - n_aux/2 - 1);
  for(int i = n_aux/2 + 1 ; i < n_aux; i++){
    tau_aux[i] = double(i - n_aux/2)*dtau;
  }
  for(int i = 0; i < n_aux; i++){
    y_aux[i] = interpolate_Newton(tau_aux[i] , &tau_values_reduced, &y_values_reduced, this->order);
  }

  aux_coeffs_y = return_coefficients(&tau_aux, &y_aux, polynomial_order);
  
  vector<vector<double> > Bell_matrix(0, vector<double>(0, 0.0));
  vector<vector<double> > inverse_Bell_matrix(0, vector<double>(0, 0.0));
  return_Bell_matrix(&Bell_matrix, &inverse_Bell_matrix, &aux_coeffs_y);
  
  (*G_kappa_coeffs) = vector<vector<double> >(bins.size(), vector<double>(polynomial_order+1, 0.0));
  
  double auxilliary_1_over_w_c = 0.5/w*inverse_c_over_e5;
  double auxilliary_1_over_w_c_F_prime = 0.5/w*inverse_c_over_e5*F_prime_at_0;
  
  /*
    TOM WAS HERE
    - if a function is used as a conditional in a loop then that function will be called at every iteration.
    In this, we can save the bin size ahead of time and get rid of calls to the size() function.
   */
  int binsize = bins.size();
  for(int b = 0; b < binsize; b++){
    
    cL = 0.0;
    dcL1 = 0.0;
    dcL2 = 0.0;
    cNL = 0.0;
    
    for(int ell = 0; ell < this->ell_max; ell++){
      cL += this->current_P_delta_L_in_trough_format_times_trough_legendres[ell]*this->bin_Legendres[b][ell];
      cNL += this->current_P_delta_NL_in_trough_format_times_trough_legendres[ell]*this->bin_Legendres[b][ell];
      dcL1 += this->current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough[ell]*this->bin_Legendres[b][ell];
      dcL2 += this->current_P_delta_L_in_trough_format_times_trough_legendres[ell]*this->dbin_Legendres_dtheta_bin[b][ell];
    }
    

    dcL1 *= R1*auxilliary_1_over_w_c_F_prime;
    dcL2 *= bins[b]*auxilliary_1_over_w_c_F_prime;
    
    dlncL1 = dcL1/cL;
    dlncL2 = dcL2/cL;
    
    double R2 = bins[b];
    double delta;
    double F;
    
    //this->set_linear_Legendres(1, R1*auxilliary_1_over_w_c);
    this->set_linear_Legendres(3, R2*auxilliary_1_over_w_c);
    
    vector<double> delta_2_values(4, 0.0);
    vector<double> tau_2_values(4, 0.0);
    
    for(int i = 0; i < n_aux; i++){
      delta = interpolate_Newton(tau_aux[i] , &tau_values_reduced, &delta_values_reduced, this->order);
      F = interpolate_Newton(tau_aux[i] , &tau_values_reduced, &F_values_reduced, this->order);
      this->set_linear_Legendres(1, R1*sqrt(1+F)/w/c_over_e5);
      double V_1 = D_growth*D_growth*interpolate_neville_aitken(log_R1 + 0.5*log(1.0+F) - log(c_over_e5), &this->log_top_hat_radii, &this->top_hat_cylinder_variances, this->order);
      
      double d_start = delta;
      if(i == 1) d_start = delta_2_values[3];
      if(i == 2) d_start = 2.0*delta_2_values[3] - delta_2_values[2];
      if(i == 3){
        double a = 0.5*(delta_2_values[3] - delta_2_values[1]);
        double b = 0.5*(delta_2_values[3]- 2.0*delta_2_values[2] + delta_2_values[1]);
        d_start = delta_2_values[2] + 2.0*a + 4.0*b;
      }
      if(i > 3){
        d_start = neville_aitken(tau_aux[i], &tau_2_values, &delta_2_values);
      }
      
      delta_2_values[0] = delta_2_values[1];
      delta_2_values[1] = delta_2_values[2];
      delta_2_values[2] = delta_2_values[3];
      
      tau_2_values[0] = tau_2_values[1];
      tau_2_values[1] = tau_2_values[2];
      tau_2_values[2] = tau_2_values[3];
      
      delta_2_values[3] = get_delta_2_from_delta_1_final(w, d_start, delta, V_1, R1*sqrt(1+F), R2, &delta_values, &F_values, this);
      tau_2_values[3] = tau_aux[i];    
      F = interpolate_neville_aitken(delta_2_values[3], &delta_values, &F_values, this->order);
      G_kappa[i] = F;
    }
    aux_coeffs_G_kappa = return_coefficients(&tau_aux, &G_kappa, polynomial_order);

      
    for(int i = 0; i < polynomial_order+1; i++){
      for(int j = 0; j <= i; j++){
        (*G_kappa_coeffs)[b][i] += inverse_Bell_matrix[i][j]*aux_coeffs_G_kappa[j]*fact[j];
      }
      (*G_kappa_coeffs)[b][i] /= fact[i];
    }
    
    
    /*
     * Rescale all coefficients to the non-linear power spectrum:
     * 
     */
    
    for(int i = 0; i < polynomial_order+1; i++){
      //double A = pow(cL, i) + double(i)*cL*pow(vL, i-1);
      //double B = pow(cNL, i) + double(i)*cNL*pow(vNL, i-1);
      double A = cL*pow(vL, i-1);
      double B = cNL*pow(vNL, i-1);
      (*G_kappa_coeffs)[b][i] *= B/A;
    }
    
    
    (*G_kappa_coeffs)[b][0] = 0.0;
    (*G_kappa_coeffs)[b][1] = cNL;
    (*G_kappa_coeffs)[b][2] = one_plus_mu*(2.0*vNL*cNL + cNL*cNL) + cNL*cNL*dlncL2 + 0.5*cNL*vNL*dlnvL_dlnR + vNL*cNL*dlncL1;
    (*G_kappa_coeffs)[b][2] *= 0.5*(1.0+this->S3_enhance);
    
    
    for(int i = 0; i < n_reduced; i++){
      (*G_kappa_of_w)[b][i] = return_polnomial_value(tau_values_reduced[i], &aux_coeffs_G_kappa);
    }
    
  }
  
  
  
  
  /*
   * 
   * <---------------------
   * 
   */
  
  (*G_coeffs) = vector<double>(polynomial_order+1, 0.0);

  
  for(int i = 0; i < n_aux; i++){
    //G[i] = interpolate_Newton(tau_aux[i] , &tau_values, &phi_values, this->order);
    G[i] = interpolate_Newton(tau_aux[i] , &tau_values_reduced, &F_values_reduced, this->order);
  }
  
  aux_coeffs_G = return_coefficients(&tau_aux, &G, polynomial_order);
  
  for(int i = 0; i < polynomial_order+1; i++){
    for(int j = 0; j <= i; j++){
      (*G_coeffs)[i] += inverse_Bell_matrix[i][j]*aux_coeffs_G[j]*fact[j];
    }
    (*G_coeffs)[i] /= fact[i];
  }
  
  
  for(int i = polynomial_order; i > 0; i--){
    (*G_coeffs)[i] = (*G_coeffs)[i-1]/double(i);
  }
    
  
  double A, B;
  
  for(int i = 0; i < polynomial_order+1; i++){
    
    if(i > 0){
      A = pow(vL, i-1);
      B = pow(vNL, i-1);
      (*G_coeffs)[i] *= B/A;
    }
    else{
      (*G_coeffs)[i] = 0.0;
    }
  }
  
  // set quadratic and cubic coefficients to our previous analytic calculations:
  (*G_coeffs)[0] = 0.0;
  (*G_coeffs)[1] = 0.0;
  (*G_coeffs)[2] = vNL*0.5;
  (*G_coeffs)[3] = vNL*vNL*(0.5*one_plus_mu + 0.25*dlnvL_dlnR)*(1.0+this->S3_enhance);
  
  
}



void Matter::return_delta_NL_of_delta_L(double eta, vector<double> *delta_L_values, vector<double> *delta_NL_values){
  (*delta_L_values) = this->delta_values;
  (*delta_NL_values) = this->delta_values;
  for(int i = 0; i < this->delta_values.size(); i++){
    (*delta_NL_values)[i] = interpolate_neville_aitken(eta, &this->eta_NL, &this->spherical_collapse_evolution_of_delta[i], this->order);
  }
}




void Matter::return_delta_NL_of_delta_L_and_dF_ddelta(double eta, vector<double> *delta_L_values, vector<double> *delta_NL_values, vector<double> *delta_NL_prime_values){
  (*delta_L_values) = this->delta_values;
  (*delta_NL_values) = this->delta_values;
  (*delta_NL_prime_values) = this->delta_values;
  for(int i = 0; i < this->delta_values.size(); i++){
    (*delta_NL_values)[i] = interpolate_neville_aitken(eta, &this->eta_NL, &this->spherical_collapse_evolution_of_delta[i], this->order);
    (*delta_NL_prime_values)[i] = interpolate_neville_aitken(eta, &this->eta_NL, &this->spherical_collapse_evolution_of_delta_ddelta[i], this->order);
  }
}



double Matter::return_Delta_prime_of_eta(double eta){
  
  return interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta_prime, this->order);
  
}
