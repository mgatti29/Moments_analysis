
double compute_variance(double th, vector<double> Cell){
  
  int n = Cell.size();
  int l_max = n-1;
  
  double Pl_th_max[l_max+1]; gsl_sf_legendre_Pl_array(l_max+1, cos(th), Pl_th_max);

  int ell;
  double var = 0.0;
  double A = constants::pi2*(1.0-cos(th));
  
  
  
  for(int ell = 1; ell < n; ell++){
    var += Cell[ell]*(Pl_th_max[ell+1]-Pl_th_max[ell-1])*(Pl_th_max[ell+1]-Pl_th_max[ell-1])/(2.0*double(ell)+1.0);
  }
  
  var *= constants::pi/(A*A);
  
  return var;
  

}




void Matter_2D::compute_PDF_and_mean_kappas_from_variable_kappa0(PDF_MODUS PDF_modus, double theta, vector<double> bins, vector<double> *delta_values, vector<double> *PDF, vector<vector<double> > *mean_deltas, vector<vector<vector<double> > > *mean_kappas, double* trough_variance, double* trough_skewness){

  this->PDF_modus = PDF_modus;
  
  if(this->PDF_modus == BERNARDEAU){
    cerr << "Bernardeau computations out of date!!!\n";
    this->compute_PDF_and_mean_kappas_from_variable_kappa0_Bernardeau(theta, bins, delta_values, PDF, mean_deltas, mean_kappas, trough_variance, trough_skewness);
    //exit(1);
  }
  else{
  
    int n_w = 150;
    int n_bin = bins.size();
    int n;
    int n_source_bins = this->source_kernels.size();
    
    double delta;
    double D_delta;
    double delta_min;
    double delta_max;  

    
    //cout << "Computing projected generating function...\n";
    set_2D_stats(n_w, theta, bins);

    double variance = this->stats_2D.variance_of_delta;
    double sigma = sqrt(variance);
    double skewness = this->stats_2D.skewness_of_delta;
    if(skewness <= 0.0){
      this->PDF_modus = GAUSSIAN;
      skewness = 0.0;
    }
    double delta_0;
    double sigma_Gauss;
    double mean_Gauss;
    (*trough_variance) = variance;
    (*trough_skewness) = skewness;
    

    D_delta = 0.1*sigma;
    if(this->PDF_modus != GAUSSIAN){
      delta_0 = this->stats_2D.d0;
      sigma_Gauss = sqrt(log(1.0 + variance/delta_0/delta_0));
      mean_Gauss = log(delta_0) - 0.5*sigma_Gauss*sigma_Gauss;
      delta_min = -delta_0;
      delta_max = exp(mean_Gauss + 5.0*sigma_Gauss) - delta_0;
      delta_max = max(delta_max, 2.0);
    }
    else{
      delta_min = -5.0*sigma;
      delta_max = -delta_min;
    }

    n = 1 + int((delta_max - delta_min)/D_delta + 0.1);

    delta_values->resize(n, 0.0);
    PDF->resize(n, 0.0);
    mean_deltas->resize(n_bin, vector<double>(n, 0.0));
    mean_kappas->resize(n_source_bins, vector<vector<double> >(n_bin, vector<double>(n, 0.0)));
  
    //cout << "Computing PDF...\n";
    for(int i = 0; i < n; i++){
      delta = delta_min + double(i)*D_delta;
      (*delta_values)[i] = delta;
      if(this->PDF_modus != GAUSSIAN){
        (*PDF)[i] = PDF_of_delta(delta, delta_0, variance);
      }
      else{
        (*PDF)[i] = exp(-0.5*delta*delta/variance)/sqrt(2.0*constants::pi*variance);
      }
      //cout << delta << setw(20);
      //cout << (*PDF)[i] << setw(20);
      //cout << exp(-0.5*delta*delta/variance)/sqrt(2.0*constants::pi*variance) << '\n';
    }
    
    double covariance;
    double triple;
    double covariance_without_source_bias;
    double triple_without_source_bias;
    
    for(int b = 0; b < n_bin; b++){
      for(int s = 0; s < n_source_bins; s++){
        this->stats_2D.return_moments(b, s, &covariance_without_source_bias, &triple_without_source_bias);
        //this->stats_2D.return_moments(b, s, &covariance, &triple);

        for(int i = 0; i < n; i++){
          delta = (*delta_values)[i];
          covariance = covariance_without_source_bias;
          triple = triple_without_source_bias;
          if(abs(this->source_biases[s]) >= eps_from_0)
            this->stats_2D.return_moments(b, s, delta, this->source_biases[s], &covariance, &triple);
          if(this->PDF_modus != GAUSSIAN){
            
            
            if(this->PDF_modus == LOGNORMAL){
              (*mean_kappas)[s][b][i] = expectation_of_kappa_given_delta_as_function_of_kdd(delta, delta_0, triple, variance, covariance);
            }
            else if(this->PDF_modus == LOGNORMAL_FIX_D0){
              (*mean_kappas)[s][b][i] = expectation_of_kappa_given_delta_as_function_of_kdd(delta, delta_0, this->stats_2D.kdd_values[s], variance, covariance, this->stats_2D.kd_values[s]);
            }
            else{
              cerr << "No valid PDF modus has been set!\n";
              cerr << "(see compute_PDF_and_mean_kappas_from_variable_kappa0(...) in Matter_2D_PDF_computation.h)\n";
              exit(1);
            }
          }
          else{
            (*mean_kappas)[s][b][i] = delta*covariance/variance;
          }
          
        }
      }
      
      
      this->stats_2D.return_moments_delta(b, &covariance, &triple);

      for(int i = 0; i < n; i++){
        delta = (*delta_values)[i];
        if(this->PDF_modus != GAUSSIAN){
          if(this->PDF_modus == LOGNORMAL){
            (*mean_deltas)[b][i] = expectation_of_kappa_given_delta_as_function_of_kdd(delta, delta_0, triple, variance, covariance);
          }
          else if(this->PDF_modus == LOGNORMAL_FIX_D0){
            (*mean_deltas)[b][i] = expectation_of_kappa_given_delta_as_function_of_kdd(delta, delta_0, skewness, variance, covariance, variance);
          }
          else{
            cerr << "No valid PDF modus has been set!\n";
            cerr << "(see compute_PDF_and_mean_kappas_from_variable_kappa0(...) in Matter_2D_PDF_computation.h)\n";
            exit(1);
          }
        }
        else{
          (*mean_deltas)[b][i] = delta*covariance/variance;
        }
      }
    }
  }
  
}







void Matter_2D::compute_projected_phi_and_G_kappas(int n_w, double theta, vector<double> bins, vector<double> *G_coeffs_projected, vector<vector<vector<double> > > *G_kappa_coeffs_projected, vector<vector<vector<double> > > *G_kappa_coeffs_overlap, vector<vector<vector<double> > > *G_source_density_coeffs, vector<double> *covariances, vector<double> *expectation_of_kdd, vector<double> *covariances_overlap, vector<double> *expectation_of_kdd_overlap, vector<double> *covariances_source_density, vector<double> *expectation_of_kdd_source_density, vector<double> *var_kappa_at_trough_radius){

  int order = 3;
  int n_kappa = this->source_kernels.size();
  
  double dw = (this->w_max - this->w_min)/double(n_w);
  double w_offset;
  double R;
  double eta;
  double weight;
  double weight_2;
  
  // IN BERNARDEAU NOTATION, I.E. LAMBDA INSTEAD OF Y, AND PHI BEING THE CUMULANT GENERATING FUNCTION:
  vector<double> w_values(0, 0.0);
  vector<double> w_weights_within_bin(0, 0.0);
  vector<double> dw_values(0, 0.0);
  vector<double> w_boundaries(0, 0.0);
  vector<double> p_of_w(0, 0.0);
  vector<double> Cell_delta(constants::ell_max, 0.0);
  vector<vector<double> > Cells_cross(n_kappa, vector<double>(constants::ell_max, 0.0));
  vector<vector<double> > Cells_kappa((n_kappa*(n_kappa+1))/2, vector<double>(constants::ell_max, 0.0));
  
  vector<double> kd_values(n_kappa, 0.0);
  vector<double> kdd_values(n_kappa, 0.0);
  vector<double> kd_values_overlap(n_kappa, 0.0);
  vector<double> kdd_values_overlap(n_kappa, 0.0);
  vector<double> kd_values_source_density(n_kappa, 0.0);
  vector<double> kdd_values_source_density(n_kappa, 0.0);
  vector<double> var_k_values(n_kappa, 0.0);
  vector<double> kk_values((n_kappa*(n_kappa+1))/2, 0.0);
  vector<double> kkd_values((n_kappa*(n_kappa+1))/2, 0.0);
  
  
  if(this->lens_kernels[0].histogram_modus == 1){
    int n = 0, steps_per_histogram_bin = 1;
    this->lens_kernels[0].return_w_boundaries_and_pofw(&w_boundaries, &p_of_w);
    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0) n++;
    }
    while(n*steps_per_histogram_bin < n_w){
      steps_per_histogram_bin++;
    }
    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0){
        dw = (w_boundaries[i+1] - w_boundaries[i])/double(steps_per_histogram_bin);
        double sum = 0.0;
        
        for(int j = 0; j < steps_per_histogram_bin; j++){
          sum += pow(w_boundaries[i]+dw*(0.5 + double(j)), 2.0);
        }
        
        sum /= double(steps_per_histogram_bin);
        
        for(int j = 0; j < steps_per_histogram_bin; j++){
          w_values.push_back(w_boundaries[i]+dw*(0.5 + double(j)));
          w_weights_within_bin.push_back(pow(w_boundaries[i]+dw*(0.5 + double(j)), 2.0)/sum);
          dw_values.push_back(dw);
        }
      }
    }
  }
  else{
    for(double w = this->w_min + 0.5*dw; w < this->w_max; w+= dw){
      w_values.push_back(w);
      dw_values.push_back(dw);
    }
  }
  
  
  
  vector<double> weight_values(w_values.size(), 0.0);
  vector<vector<double> > weight_values_kappa(n_kappa, vector<double>(w_values.size(), 0.0));
  vector<vector<double> > weight_values_kappa_overlap(n_kappa, vector<double>(w_values.size(), 0.0));
  vector<vector<double> > weight_values_source_density(n_kappa, vector<double>(w_values.size(), 0.0));
  
  vector<vector<double> > G_coeffs_at_w(w_values.size(), vector<double>(order+1, 0.0));
  vector<vector<vector<double> > > G_kappa_coeffs_at_w(w_values.size(), vector<vector<double> >(bins.size(), vector<double>(order+1, 0.0)));

  vector<double> physical_bins = bins;

  
  for(int i = 0; i < w_values.size(); i++){
    
    double w = w_values[i];
//    cout << "\rProjecting 3D statistics along co-moving distance: " << w << "/" << w_values[w_values.size()-1] << "     ";
//    cout.flush();
    
    eta = this->universe->eta_at_a(1.0) - w;
    R = w*c_over_e5*theta;
    
    for(int b = 0; b < physical_bins.size(); b++){
      physical_bins[b] = bins[b]*w*c_over_e5;
    }
    
    this->matter->compute_lognormal_final(R, eta, &physical_bins, &G_coeffs_at_w[i], &G_kappa_coeffs_at_w[i]);
    //this->matter->compute_lognormal_old(R, eta, w*c_over_e5*bins, &G_coeffs_at_w[i], &G_kappa_coeffs_at_w[i]);

    weight = this->lens_kernels[0].weight_at_comoving_distance(w)*w;
    weight_values[i] = weight;
    
    
    
    for(int l = 0; l < n_kappa; l++){
      weight = this->source_kernels[l].weight_at_comoving_distance(w)*w;
      weight_values_kappa[l][i] = weight;
      weight = this->source_kernels_overlap[l].weight_at_comoving_distance(w)*w;
      weight_values_kappa_overlap[l][i] = weight;
      weight = this->source_kernels_density[l].weight_at_comoving_distance(w)*w;
      weight_values_source_density[l][i] = weight;
      kd_values[l] += dw_values[i]*weight_values_kappa[l][i]*weight_values[i]*G_coeffs_at_w[i][2]*2.0;
      kdd_values[l] += dw_values[i]*weight_values_kappa[l][i]*weight_values[i]*weight_values[i]*G_coeffs_at_w[i][3]*6.0;
      kd_values_overlap[l] += dw_values[i]*weight_values_kappa_overlap[l][i]*weight_values[i]*G_coeffs_at_w[i][2]*2.0;
      kdd_values_overlap[l] += dw_values[i]*weight_values_kappa_overlap[l][i]*weight_values[i]*weight_values[i]*G_coeffs_at_w[i][3]*6.0;
      kd_values_source_density[l] += dw_values[i]*weight_values_source_density[l][i]*weight_values[i]*G_coeffs_at_w[i][2]*2.0;
      kdd_values_source_density[l] += dw_values[i]*weight_values_source_density[l][i]*weight_values[i]*weight_values[i]*G_coeffs_at_w[i][3]*6.0;
      
      var_k_values[l] += dw_values[i]*weight_values_kappa[l][i]*weight_values_kappa[l][i]*G_coeffs_at_w[i][2]*2.0;
    }
    
    
    this->matter->update_Cell(dw_values[i], weight_values[i], weight_values[i], &Cell_delta);
    /*int kappa_index = 0;
    for(int l = 0; l < n_kappa; l++){
      weight = this->source_kernels[l].weight_at_comoving_distance(w)*w;
      kd_values[l] += dw_values[i]*weight*weight_values[i]*G_coeffs_at_w[i][2]*2.0;
      kdd_values[l] += dw_values[i]*weight*weight_values[i]*weight_values[i]*G_coeffs_at_w[i][3]*6.0;
      this->matter->update_Cell(dw_values[i], weight_values[i], weight, &Cells_cross[l]);
      for(int m = l; m < n_kappa; m++){
        weight_2 = this->source_kernels[m].weight_at_comoving_distance(w)*w;
        kk_values[kappa_index] += dw_values[i]*weight*weight_2*G_coeffs_at_w[i][2]*2.0;
        kkd_values[kappa_index] += dw_values[i]*weight*weight_2*weight_values[i]*G_coeffs_at_w[i][3]*6.0;
        this->matter->update_Cell(dw_values[i], weight, weight_2, &Cells_kappa[kappa_index]);
        kappa_index++;
      }
    }
    */
    
  }
  
//  cout << "\rProjecting 3D statistics along co-moving distance: " << w_values[w_values.size()-1] << "/" << w_values[w_values.size()-1] << "     \n";
  
  G_coeffs_projected->resize(order+1, 0.0);
  G_kappa_coeffs_projected->resize(n_kappa, vector<vector<double> >(bins.size(), vector<double>(order+1, 0.0)));
  G_kappa_coeffs_overlap->resize(n_kappa, vector<vector<double> >(bins.size(), vector<double>(order+1, 0.0)));
  G_source_density_coeffs->resize(n_kappa, vector<vector<double> >(bins.size(), vector<double>(order+1, 0.0)));
  
  for(int j = 0; j <= order; j++){
    for(int i = 0; i < w_values.size(); i++){
      (*G_coeffs_projected)[j] += dw_values[i]*pow(weight_values[i], j)*G_coeffs_at_w[i][j];
      for(int b = 0; b < bins.size(); b++){
        for(int s = 0; s < n_kappa; s++){
          (*G_kappa_coeffs_projected)[s][b][j] += dw_values[i]*weight_values_kappa[s][i]*pow(weight_values[i], j)*G_kappa_coeffs_at_w[i][b][j];
          (*G_kappa_coeffs_overlap)[s][b][j] += dw_values[i]*weight_values_kappa_overlap[s][i]*pow(weight_values[i], j)*G_kappa_coeffs_at_w[i][b][j];
          (*G_source_density_coeffs)[s][b][j] += dw_values[i]*weight_values_source_density[s][i]*pow(weight_values[i], j)*G_kappa_coeffs_at_w[i][b][j];
        }
      }
    }
  }
  
  double var = (*G_coeffs_projected)[2]*2.0;
  double skew = (*G_coeffs_projected)[3]*6.0;
  double d0 = get_delta0(var, skew);
  double cov = 0.0;
  double triple = 0.0;
  

  
  int index = find_index(20.0*constants::arcmin, &bins);
  
  vector<vector<double> > Gaussian_deltaS_of_w_given_deltaL_equals_1(w_values.size(), vector<double>(bins.size(), 0.0));
  vector<vector<double> > lognormal_deltaS_of_w_given_deltaL_equals_1(w_values.size(), vector<double>(bins.size(), 0.0));
  vector<double> Gaussian_deltaS_given_deltaL_equals_1(bins.size(), 0.0);
  vector<double> lognormal_deltaS_given_deltaL_equals_1(bins.size(), 0.0);
  
  for(int i = 0; i < w_values.size(); i++){
    for(int b = 0; b < bins.size(); b++){
    
      cov = weight_values[i]*2.0*G_coeffs_at_w[i][2];
      triple = weight_values[i]*weight_values[i]*6.0*G_coeffs_at_w[i][3];
      Gaussian_deltaS_of_w_given_deltaL_equals_1[i][b] = 1.0*cov/var;
      
      //if(b == index){
      //  cout << w_values[i] << setw(20) << triple - 2.0*var*cov/d0 << '\n';
      //}
      
    }
  }
  
  (*covariances) = kd_values;
  (*expectation_of_kdd) = kdd_values;
  (*covariances_overlap) = kd_values_overlap;
  (*expectation_of_kdd_overlap) = kdd_values_overlap;
  (*covariances_source_density) = kd_values_source_density;
  (*expectation_of_kdd_source_density) = kdd_values_source_density;
  (*var_kappa_at_trough_radius) = var_k_values;
  
  //this->initialize_2D_stats((*G_coeffs_projected)[2]*2.0, (*G_coeffs_projected)[3]*6.0, kd_values, kdd_values, kk_values, kkd_values, Cell_delta, Cells_cross, Cells_kappa);
  
  cout << "#ell" << setw(20) << "C_ell" << '\n';
  for(int i = 0; i < Cell_delta.size(); i++){
    cout << i << setw(20) << Cell_delta[i] << '\n';
  }
  
}


void Matter_2D::compute_projected_phi_and_G_kappas_Bernardeau(int n_w, double theta, vector<double> bins, vector<double> *G_coeffs_projected, vector<vector<double> > *G_delta_coeffs_projected, vector<vector<vector<double> > > *G_kappa_coeffs_projected, vector<double> *lambda_values, vector<double> *phi_lambda_values, vector<double> *phi_prime_values, vector<vector<double> > *G_delta_values, vector<vector<vector<double> > > *G_kappa_values, double *kd, double *kdd, double *kkd, double *var_kappa_at_trough_radius){

  int n_lambda, n_aux, order = constants::coeff_order;
  int n_kappa = this->source_kernels.size();
  
  double dw = (this->w_max - this->w_min)/double(n_w);
  double R;
  double eta;
  double weight;
  double lambda_min = 0.0;
  double lambda_max = 0.0;
  double dlambda = 0.2;
  
  // IN BERNARDEAU NOTATION, I.E. LAMBDA INSTEAD OF Y, AND PHI BEING THE CUMULANT GENERATING FUNCTION:
  vector<double> lambda_of_w;
  vector<double> phi_lambda_of_w;
  vector<double> phi_prime_lambda_of_w;
  vector<vector<double> > G_kappa_of_w;
  vector<double> G_coeffs;
  vector<vector<double> > lambda_values_at_w(0);
  vector<vector<double> > phi_values_at_w(0);
  vector<vector<double> > phi_prime_values_at_w(0);
  vector<vector<vector<double> > > G_kappa_at_w(0);
  
    
  // IN BERNARDEAU NOTATION, I.E. LAMBDA INSTEAD OF Y, AND PHI BEING THE CUMULANT GENERATING FUNCTION:
  vector<double> w_values(0, 0.0);
  vector<double> w_weights_within_bin(0, 0.0);
  vector<double> dw_values(0, 0.0);
  vector<double> w_boundaries(0, 0.0);
  vector<double> p_of_w(0, 0.0);
  
  
  if(this->lens_kernels[0].histogram_modus == 1){
    int n = 0, steps_per_histogram_bin = 1;
    this->lens_kernels[0].return_w_boundaries_and_pofw(&w_boundaries, &p_of_w);
    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0) n++;
    }
    while(n*steps_per_histogram_bin < n_w){
      steps_per_histogram_bin++;
    }
    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0){
        dw = (w_boundaries[i+1] - w_boundaries[i])/double(steps_per_histogram_bin);
        double sum = 0.0;
        
        for(int j = 0; j < steps_per_histogram_bin; j++){
          sum += pow(w_boundaries[i]+dw*(0.5 + double(j)), 2.0);
        }
        
        sum /= double(steps_per_histogram_bin);
        
        for(int j = 0; j < steps_per_histogram_bin; j++){
          w_values.push_back(w_boundaries[i]+dw*(0.5 + double(j)));
          w_weights_within_bin.push_back(pow(w_boundaries[i]+dw*(0.5 + double(j)), 2.0)/sum);
          dw_values.push_back(dw);
        }
      }
    }
  }
  else{
    for(double w = this->w_min + 0.5*dw; w < this->w_max; w+= dw){
      w_values.push_back(w);
      dw_values.push_back(dw);
    }
  }
  
  
  vector<double> weight_values(w_values.size(), 0.0);
  vector<vector<double> > weight_values_kappa(n_kappa, vector<double>(w_values.size(), 0.0));
  vector<vector<double> > G_coeffs_at_w(w_values.size(), vector<double>(order+1, 0.0));
  vector<vector<vector<double> > > G_kappa_coeffs_at_w(w_values.size(), vector<vector<double> >(bins.size(), vector<double>(order+1, 0.0)));
  vector<double> physical_bins = bins;
  
  (*kd) = 0.0;
  (*kdd) = 0.0;
  (*kkd) = 0.0;
  (*var_kappa_at_trough_radius) = 0.0;
  
  for(int i = 0; i < w_values.size(); i++){
    
    double w = w_values[i];
    cout << '\r' << w << "/" << this->w_max << "     ";
    cout.flush();
    eta = this->universe->eta_at_a(1.0) - w;
    R = w*c_over_e5*theta;
    
    for(int b = 0; b < physical_bins.size(); b++){
      physical_bins[b] = w*c_over_e5*bins[b];
    }
    
    this->matter->compute_Bernardeau_final(R, eta, physical_bins, &G_coeffs_at_w[i], &G_kappa_coeffs_at_w[i], &lambda_of_w, &phi_lambda_of_w, &phi_prime_lambda_of_w, &G_kappa_of_w);
    weight = this->lens_kernels[0].weight_at_comoving_distance(w)*w;

    weight_values[i] = weight;
    lambda_values_at_w.push_back(lambda_of_w);
    phi_values_at_w.push_back(phi_lambda_of_w);
    phi_prime_values_at_w.push_back(phi_prime_lambda_of_w);
    G_kappa_at_w.push_back(G_kappa_of_w);

    n_lambda = lambda_of_w.size();
    
    if(i == 0){
      lambda_max = lambda_of_w[n_lambda - 1]/weight;
      lambda_min = lambda_of_w[0]/weight;
    }
    else{
      if(lambda_of_w[n_lambda - 1]/weight < lambda_max) lambda_max = lambda_of_w[n_lambda - 1]/weight;
      if(lambda_of_w[0]/weight > lambda_min) lambda_min = lambda_of_w[0]/weight;
    }
    
    for(int s = 0; s < n_kappa; s++){
      weight = this->source_kernels[s].weight_at_comoving_distance(w)*w;
      weight_values_kappa[s][i] = weight;
    }
    
  }  
  cout << '\r' << this->w_max << "/" << this->w_max << "     \n";
  
  (*G_coeffs_projected) = vector<double>(order+1, 0.0);
  (*G_delta_coeffs_projected) = vector<vector<double> >(bins.size(), vector<double>(order+1, 0.0));
  (*G_kappa_coeffs_projected) = vector<vector<vector<double> > >(n_kappa, vector<vector<double> >(bins.size(), vector<double>(order+1, 0.0)));
  
  for(int j = 0; j <= order; j++){
    for(int i = 0; i < w_values.size(); i++){
      (*G_coeffs_projected)[j] += dw_values[i]*pow(weight_values[i], j)*G_coeffs_at_w[i][j];
      if(j == 2){
        (*kd) += dw_values[i]*weight_values_kappa[0][i]*weight_values[i]*G_coeffs_at_w[i][j];
        (*var_kappa_at_trough_radius) += dw_values[i]*weight_values_kappa[0][i]*weight_values_kappa[0][i]*G_coeffs_at_w[i][j];
      }
      if(j == 3){
        (*kdd) += dw_values[i]*weight_values_kappa[0][i]*weight_values[i]*weight_values[i]*G_coeffs_at_w[i][j];
        (*kkd) += dw_values[i]*weight_values_kappa[0][i]*weight_values_kappa[0][i]*weight_values[i]*G_coeffs_at_w[i][j];
      }
      
      for(int b = 0; b < bins.size(); b++)
        (*G_delta_coeffs_projected)[b][j] += dw_values[i]*pow(weight_values[i], j+1)*G_kappa_coeffs_at_w[i][b][j];
      
      for(int s = 0; s < n_kappa; s++)
        for(int b = 0; b < bins.size(); b++)
          (*G_kappa_coeffs_projected)[s][b][j] += dw_values[i]*weight_values_kappa[s][i]*pow(weight_values[i], j)*G_kappa_coeffs_at_w[i][b][j];
    }
  }
  (*kd) *= 2.0;
  (*var_kappa_at_trough_radius) *= 2.0;
  (*kdd) *= 6.0;
  (*kkd) *= 6.0;
  
  double variance = 2.0*(*G_coeffs_projected)[2];
  double skewness = 6.0*(*G_coeffs_projected)[3];
  double delta_min = -get_delta0(variance, skewness);
  
  
  cout << "        delta_0: " << get_delta0(variance, skewness) << '\n';
  cout << "          Var_T: " << variance << '\n';
  cout << "         Skew_T: " << skewness << '\n';

  n_aux = int(lambda_max/dlambda + 0.1);
  lambda_max = double(n_aux)*dlambda;
  
  int index = 0;
  (*G_delta_values) = vector<vector<double> >(bins.size(), vector<double>(0, 0.0));
  (*G_kappa_values) = vector<vector<vector<double> > >(n_kappa, vector<vector<double> >(bins.size(), vector<double>(0, 0.0)));
  (*phi_lambda_values) = vector<double>(0, 0.0);
  (*phi_prime_values) = vector<double>(0, 0.0);
  do{
    lambda_values->emplace(lambda_values->begin(), lambda_max - double(index)*dlambda);
    phi_lambda_values->emplace(phi_lambda_values->begin(), 0.0);
    phi_prime_values->emplace(phi_prime_values->begin(), 0.0);
    
    for(int i = 0; i < w_values.size(); i++){
      (*phi_lambda_values)[0] += dw_values[i]*interpolate_Newton((*lambda_values)[0]*weight_values[i], &lambda_values_at_w[i], &phi_values_at_w[i], this->order);
      (*phi_prime_values)[0] += dw_values[i]*weight_values[i]*interpolate_Newton((*lambda_values)[0]*weight_values[i], &lambda_values_at_w[i], &phi_prime_values_at_w[i], this->order);
      //(*phi_prime_values)[0] += dw_values[i]*weight_values[i]*interpolate_neville_aitken_derivative((*lambda_values)[0]*weight_values[i], &lambda_values_at_w[i], &phi_values_at_w[i], this->order);
    }
    
    for(int b = 0; b < bins.size(); b++){
      (*G_delta_values)[b].emplace((*G_delta_values)[b].begin(), 0.0);
      for(int i = 0; i < w_values.size(); i++){
        (*G_delta_values)[b][0] += dw_values[i]*weight_values[i]*interpolate_Newton((*lambda_values)[0]*weight_values[i], &lambda_values_at_w[i], &G_kappa_at_w[i][b], this->order);
      }
    }
    
    for(int s = 0; s < n_kappa; s++){
      for(int b = 0; b < bins.size(); b++){
        (*G_kappa_values)[s][b].emplace((*G_kappa_values)[s][b].begin(), 0.0);
        for(int i = 0; i < w_values.size(); i++){
          (*G_kappa_values)[s][b][0] += dw_values[i]*weight_values_kappa[s][i]*interpolate_Newton((*lambda_values)[0]*weight_values[i], &lambda_values_at_w[i], &G_kappa_at_w[i][b], this->order);
        }
      }
    }
    
    
    index++;
    cout << '\r' << (*phi_prime_values)[0] << "/" << delta_min << "     ";
    cout.flush();
  }while((*phi_prime_values)[0] >= delta_min && (*lambda_values)[0] > lambda_min);
  
  cout << '\r' << delta_min << "/" << delta_min << "     \n";

}

void Matter_2D::initialize_2D_stats(double dd, double ddd, vector<double> kd_values, vector<double> kdd_values, vector<double> kk_values, vector<double> kkd_values, vector<double> Cl_delta, vector<vector<double> > Cls_cross, vector<vector<double> > Cls_kappa){
  
  int n = Cl_delta.size();
  int n_kappa = this->source_kernels.size();
  
  this->stats_2D.variance_of_delta = dd;
  this->stats_2D.skewness_of_delta = ddd;
  
  this->stats_2D.kd_values = kd_values;
  this->stats_2D.kdd_values = kdd_values;
  this->stats_2D.kkd_values = kkd_values;
  this->stats_2D.kk_values_total = kk_values;
  this->stats_2D.kkd_values = kkd_values;
  
  this->stats_2D.Cell_delta = Cl_delta;
  this->stats_2D.Cells_cross = Cls_cross;
  this->stats_2D.Cells_kappa = Cls_kappa;
  this->stats_2D.Cells_kappa_uncorrelated = vector<vector<double> >((n_kappa*(n_kappa+1))/2, vector<double>(n, 0.0));
  
  this->stats_2D.d0 = get_delta0(dd, ddd);
  
  for(int i = 0; i < n_kappa; i++){
    this->stats_2D.k0_values.push_back(get_kappa0(this->stats_2D.d0, dd, kd_values[i], kdd_values[i]));
  }
  
  int kappa_index = 0;
  for(int i = 0; i < n_kappa; i++){    
    for(int j = i; j < n_kappa; j++){
      double d0 = this->stats_2D.d0;
      double k01 = this->stats_2D.k0_values[i];
      double k02 = this->stats_2D.k0_values[j];
      double dk1 = this->stats_2D.kd_values[i];
      double dk2 = this->stats_2D.kd_values[j];
      double dk1k2 = this->stats_2D.kkd_values[kappa_index];
      this->stats_2D.kk_values_correlated.push_back(get_cov_of_two_kappas(d0, k01, k02, dk1, dk2, dk1k2));
      this->stats_2D.kk_values_uncorrelated.push_back(this->stats_2D.kk_values_total[kappa_index] - this->stats_2D.kk_values_correlated[kappa_index]);
      kappa_index++;
    }
  }


  this->stats_2D.initialized = 1;

}


void Matter_2D::get_total_kappa_Cells(vector<vector<double> > * Cls_kappa){
  
  int n_kappa = this->source_kernels.size();
  
  double weight;
  double weight_2;
  double dw;
  
  vector<double> w_values(0, 0.0);
  vector<double> dw_values(0, 0.0);
  vector<double> w_boundaries(0, 0.0);
  vector<double> p_of_w(0, 0.0);

      
  int n = len_ww;
  
  double w_max = this->source_kernels[0].return_w_max();
  double w_min = this->source_kernels[0].return_w_min();
  
  for(int i = 1; i < n_kappa; i++){
    if(w_max < this->source_kernels[i].return_w_max())
      w_max = this->source_kernels[i].return_w_max();
    if(w_min > this->source_kernels[i].return_w_min())
      w_min = this->source_kernels[i].return_w_min();
  }
      
  dw = (w_max - w_min)/double(n);
      
  for(int i = 0; i < n; i++){
    w_values.push_back(w_min+dw*(0.5+double(i)));
    dw_values.push_back(dw);
  }


  for(int i = 0; i < w_values.size(); i++){
    
    double w = w_values[i];
    cout <<"boia " << endl;
    cout << "\rProjecting 2D convergence power spectrum: " << w << "/" << w_values[w_values.size()-1] << "     ";
    cout.flush();

    
    cout << this->source_kernels[0].weight_at_comoving_distance(w)<<endl;
      
    weight = this->source_kernels[0].weight_at_comoving_distance(w)*w;

    this->matter->update_Cell(w, dw_values[i], weight, weight, &(*Cls_kappa)[0]);
    
    int kappa_index = 0;
    for(int k = 0; k < n_kappa; k++){
      weight = this->source_kernels[k].weight_at_comoving_distance(w)*w;
      for(int l = k; l < n_kappa; l++){
        if(kappa_index != 0){
          weight_2 = this->source_kernels[l].weight_at_comoving_distance(w)*w;
          this->matter->update_Cell(dw_values[i], weight, weight_2, &(*Cls_kappa)[kappa_index]);
        }
        kappa_index++;
      }
    }
        
  }  
  cout << "\rProjecting 2D convergence power spectrum: " << w_values[w_values.size()-1] << "/" << w_values[w_values.size()-1] << "     \n";
  
}


//Marco was here
void Matter_2D::get_total_kappa_Cells_lens(vector<vector<double> > * Cls_kappa){

  int n_kappa = this->lens_kernels.size();

  double weight;
  double weight_2;
  double dw;

  vector<double> w_values(0, 0.0);
  vector<double> dw_values(0, 0.0);
  vector<double> w_boundaries(0, 0.0);
  vector<double> p_of_w(0, 0.0);

  int n = len_ww;
  cout << len_ww<<endl;
  double w_max = this->lens_kernels[0].return_w_max();
  double w_min = this->lens_kernels[0].return_w_min();

  for(int i = 1; i < n_kappa; i++){
    if(w_max < this->lens_kernels[i].return_w_max())
      w_max = this->lens_kernels[i].return_w_max();
    if(w_min > this->lens_kernels[i].return_w_min())
      w_min = this->lens_kernels[i].return_w_min();
  }


  dw = (w_max - w_min)/double(n);

  for(int i = 0; i < n; i++){
    w_values.push_back(w_min+dw*(0.5+double(i)));
    dw_values.push_back(dw);
  }
  for(int jj = 0; jj < this->len_bins2; jj++){
     if(this->sd_lensing_A ==0 or this->sd_lensing_A ==1){
            this->lens_kernels[this->bins2[2*jj]-1].shear_or_delta = this->sd_lensing_A;

        }
     if(this->sd_lensing_B ==0 or this->sd_lensing_B ==1){
            this->lens_kernels[this->bins2[2*jj+1]-1].shear_or_delta = this->sd_lensing_B;
   }
   }

  //vector<vector<double> > Cells_mute((w_values.size()), vector<double>(n, 0.0));

  for(int i = 0; i < w_values.size(); i++){

    double w = w_values[i];

    cout << "\rProjecting 2D convergence power spectrum: " << w << "/" << w_values[w_values.size()-1] << "     ";
    cout.flush();


    weight_2 = this->lens_kernels[this->bins2[0]-1].weight_at_comoving_distance(w)*w;  //Marco was here
    weight = this->lens_kernels[this->bins2[1]-1].weight_at_comoving_distance(w)*w;


    this->matter->update_Cell(w, dw_values[i], weight, weight_2, &(*Cls_kappa)[0]);


    //**************************************************************************
    //this->matter->update_Cell(w, dw_values[i], weight, weight_2, &(Cells_mute)[i]);



    /************************************************************************
    /*
    int kappa_index = 0;
    for(int k = 0; k < n_kappa; k++){
      weight = this->lens_kernels[k].weight_at_comoving_distance(w)*w;

      for(int l = k; l < n_kappa; l++){
        if(kappa_index != 0){
          weight_2 = this->lens_kernels_2[l].weight_at_comoving_distance(w)*w;
          this->matter->update_Cell(dw_values[i], weight, weight_2, &(*Cls_kappa)[kappa_index]); //routine in matter.h

        }
        kappa_index++;
      }
    }
    */

  //this->stats_2D.Cells_kappa_total = (*Cls_kappa);
  }
  //save_output:
   //save file *************************************************

   int count = 0;
    //saving output by output and total *************


  for (int jj =0; jj< len_bins2; jj++){

    remove(this->save_out[count]);
    FILE* F = fopen("./Cl.txt" , "w");
    fclose(F);
    fstream out1;
    out1.open(save_out[count]);
    for(int sm = 0; sm <10000; sm++){
        out1 << setprecision(2);
        out1 << sm << setw(15);
        out1 << scientific << setprecision(5);
        out1 <<(*Cls_kappa)[jj][sm] << setw(15);
        //for(int i = 0; i < w_values.size(); i++){
        //    out1 <<(Cells_mute)[i][sm] << setw(15);
       // }
        out1 <<"\n";

      }
    count +=1;
    out1.close();
    }

  cout << "\rProjecting 2D convergence power spectrum: " << w_values[w_values.size()-1] << "/" << w_values[w_values.size()-1] << "     \n";

}




void Matter_2D::set_2D_stats(int n_w, double theta, vector<double> bins){

  this->matter->set_permanent_Legendres(constants::ell_max, theta, bins);
  this->stats_2D.uni = this->universe;
  this->stats_2D.cosmo = this->cosmology;
  this->stats_2D.bins = bins;
  
  int order = 3;
  int n_kappa = this->source_kernels.size();
  
  double dw = (this->w_max - this->w_min)/double(n_w);
  double R;
  double eta;
  double weight, weight2;

  // IN BERNARDEAU NOTATION, I.E. LAMBDA INSTEAD OF Y, AND PHI BEING THE CUMULANT GENERATING FUNCTION:
  this->stats_2D.w_values = vector<double>(0, 0.0);
  this->stats_2D.dw_values = vector<double>(0, 0.0);
  
  vector<double> w_boundaries(0, 0.0);
  vector<double> p_of_w(0, 0.0);
  
  this->stats_2D.Cell_delta = vector<double>(constants::ell_max, 0.0);
  this->stats_2D.Cells_cross = vector<vector<double> >(n_kappa, vector<double>(constants::ell_max, 0.0));
  this->stats_2D.Cells_kappa = vector<vector<double> >((n_kappa*(n_kappa+1))/2, vector<double>(constants::ell_max, 0.0));

  this->stats_2D.Cell_delta = vector<double> (constants::ell_max, 0.0); //Marco here

  this->stats_2D.kd_values = vector<double>(n_kappa, 0.0);
  this->stats_2D.kdd_values = vector<double>(n_kappa, 0.0);
  this->stats_2D.kk_values_total = vector<double>((n_kappa*(n_kappa+1))/2, 0.0);
  this->stats_2D.kkd_values = vector<double>((n_kappa*(n_kappa+1))/2, 0.0);
  
  
  if(this->lens_kernels[0].histogram_modus == 1){
    int n = 0, steps_per_histogram_bin = 1;
    this->lens_kernels[0].return_w_boundaries_and_pofw(&w_boundaries, &p_of_w);
    cout<< p_of_w.size()<<endl;
    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0) n++;
    }
    while(n*steps_per_histogram_bin < n_w){
      steps_per_histogram_bin++;
    }
    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0){
        dw = (w_boundaries[i+1] - w_boundaries[i])/double(steps_per_histogram_bin);
        double sum = 0.0;
        
        for(int j = 0; j < steps_per_histogram_bin; j++){
          sum += pow(w_boundaries[i]+dw*(0.5 + double(j)), 2.0);
        }
        
        sum /= double(steps_per_histogram_bin);
        
        for(int j = 0; j < steps_per_histogram_bin; j++){
          this->stats_2D.w_values.push_back(w_boundaries[i]+dw*(0.5 + double(j)));
          this->stats_2D.dw_values.push_back(dw);
        }
      }
    }
  }
  else{
    for(double w = this->w_min + 0.5*dw; w < this->w_max; w+= dw){
      this->stats_2D.w_values.push_back(w);
      this->stats_2D.dw_values.push_back(dw);
    }
  }

  this->stats_2D.weight_values = vector<double>(this->stats_2D.w_values.size(), 0.0);
  this->stats_2D.one_over_a_values = vector<double>(this->stats_2D.w_values.size(), 0.0);
  this->stats_2D.one_over_w_values = vector<double>(this->stats_2D.w_values.size(), 0.0);
  this->stats_2D.new_weights = vector<double>(this->stats_2D.w_values.size(), 0.0);
  this->stats_2D.lognormal_deltaS_given_deltaL = vector<double>(this->stats_2D.w_values.size(), 0.0);
  this->stats_2D.weight_values_kappa = vector<vector<double> >(n_kappa, vector<double>(this->stats_2D.w_values.size(), 0.0));
  this->stats_2D.weight_values_source_density = vector<vector<double> >(n_kappa, vector<double>(this->stats_2D.w_values.size(), 0.0));
  
  this->stats_2D.G_coeffs_at_w = vector<vector<double> >(this->stats_2D.w_values.size(), vector<double>(order+1, 0.0));
  this->stats_2D.G_kappa_coeffs_at_w = vector<vector<vector<double> > >(this->stats_2D.w_values.size(), vector<vector<double> >(bins.size(), vector<double>(order+1, 0.0)));

  vector<double> physical_bins = bins;
      
  this->stats_2D.variance_of_delta = 0.0;
  this->stats_2D.skewness_of_delta = 0.0;
  
  for(int i = 0; i < this->stats_2D.w_values.size(); i++){
    
    double w = this->stats_2D.w_values[i];
    //cout << "\rProjecting 3D statistics along co-moving distance: " << w << "/" << this->stats_2D.w_values[this->stats_2D.w_values.size()-1] << "     ";
    //cout.flush();
    
    eta = this->universe->eta_at_a(1.0) - w;
    this->stats_2D.one_over_a_values[i] = 1.0/this->universe->a_at_eta(eta);
    this->stats_2D.one_over_w_values[i] = 1.0/w;
    R = w*c_over_e5*theta;
    
    for(int b = 0; b < physical_bins.size(); b++){
      physical_bins[b] = bins[b]*w*c_over_e5;
    }

    this->matter->compute_lognormal_final(R, eta, &physical_bins, &this->stats_2D.G_coeffs_at_w[i], &this->stats_2D.G_kappa_coeffs_at_w[i]);

    weight = this->lens_kernels[0].weight_at_comoving_distance(w)*w;
    this->stats_2D.weight_values[i] = weight;
    
    double dw = this->stats_2D.dw_values[i];
    
    for(int l = 0; l < n_kappa; l++){
      weight = this->source_kernels[l].weight_at_comoving_distance(w)*w;
      this->stats_2D.weight_values_kappa[l][i] = weight;
      weight = this->source_kernels_density[l].weight_at_comoving_distance(w)*w;
      this->stats_2D.weight_values_source_density[l][i] = weight;
      this->stats_2D.kd_values[l] += this->stats_2D.dw_values[i]*this->stats_2D.weight_values_kappa[l][i]*this->stats_2D.weight_values[i]*this->stats_2D.G_coeffs_at_w[i][2]*2.0;
      this->stats_2D.kdd_values[l] += this->stats_2D.dw_values[i]*this->stats_2D.weight_values_kappa[l][i]*this->stats_2D.weight_values[i]*this->stats_2D.weight_values[i]*this->stats_2D.G_coeffs_at_w[i][3]*6.0;
      this->stats_2D.kkd_values[l] += this->stats_2D.dw_values[i]*this->stats_2D.weight_values_kappa[l][i]*this->stats_2D.weight_values_kappa[l][i]*this->stats_2D.weight_values[i]*this->stats_2D.G_coeffs_at_w[i][3]*6.0;
      this->stats_2D.kk_values_total[l] += this->stats_2D.dw_values[i]*this->stats_2D.weight_values_kappa[l][i]*this->stats_2D.weight_values_kappa[l][i]*this->stats_2D.G_coeffs_at_w[i][2]*2.0;
    }
    
    
    
    this->matter->update_Cell(this->stats_2D.dw_values[i], this->stats_2D.weight_values[i], this->stats_2D.weight_values[i], &this->stats_2D.Cell_delta);
    int kappa_index = 0;
    for(int l = 0; l < n_kappa; l++){
      weight = this->source_kernels[l].weight_at_comoving_distance(w)*w;
      this->matter->update_Cell(this->stats_2D.dw_values[i], this->stats_2D.weight_values[i], weight, &this->stats_2D.Cells_cross[l]);
      for(int m = l; m < n_kappa; m++){
        weight2 = this->source_kernels[m].weight_at_comoving_distance(w)*w;
        this->matter->update_Cell(this->stats_2D.dw_values[i], weight, weight2, &this->stats_2D.Cells_kappa[kappa_index]);
        kappa_index++;
      }
    }
    
    this->stats_2D.variance_of_delta += this->stats_2D.dw_values[i]*pow(this->stats_2D.weight_values[i], 2)*this->stats_2D.G_coeffs_at_w[i][2]*2.0;
    this->stats_2D.skewness_of_delta += this->stats_2D.dw_values[i]*pow(this->stats_2D.weight_values[i], 3)*this->stats_2D.G_coeffs_at_w[i][3]*6.0;
  }
  
  this->stats_2D.d0 = get_delta0(this->stats_2D.variance_of_delta, this->stats_2D.skewness_of_delta);
  
  this->stats_2D.one_over_d0 = 1.0/this->stats_2D.d0;
  this->stats_2D.Gaussian_variance = log(1.0 + this->stats_2D.variance_of_delta/this->stats_2D.d0/this->stats_2D.d0);
  this->stats_2D.one_over_Gaussian_variance = 1.0/this->stats_2D.Gaussian_variance;
  
  this->stats_2D.set_source_bias_simplifications();
  
  this->stats_2D.initialized = 1;
  
  
  /*cout << "#ell" << setw(20) << "C_ell" << '\n';
  for(int i = 0; i < this->stats_2D.Cell_delta.size(); i++){
    cout << i << setw(20) << this->stats_2D.Cell_delta[i] << '\n';
  }*/
  /*
  cout << "#theta_min" << setw(20) << "theta_max" << setw(20) << "w" << '\n';
  for(int i = 0; i < 24; i++){
    double theta_min = 5.0*constants::arcmin*pow(600.0/5.0, double(i)/double(24));
    double theta_max = 5.0*constants::arcmin*pow(600.0/5.0, double(i+1)/double(24));
    cout << theta_min/constants::arcmin << setw(20);
    cout << theta_max/constants::arcmin << setw(20);
    cout << this->stats_2D.dd_2pt_function(theta_min, theta_max)*pow(1.54, 2) << '\n';
  }*/
    
}




void projection_stats::return_moments_delta(int theta_bin, double *covariance, double *kdd){
  
  (*covariance) = 0.0;
  (*kdd) = 0.0;

  for(int i = 0; i < this->w_values.size(); i++){
      (*covariance) += this->dw_values[i]*this->weight_values[i]*this->weight_values[i]*this->G_kappa_coeffs_at_w[i][theta_bin][1];
      (*kdd) += this->dw_values[i]*this->weight_values[i]*this->weight_values[i]*this->weight_values[i]*2.0*this->G_kappa_coeffs_at_w[i][theta_bin][2];
  }
  
}




void projection_stats::return_moments(int theta_bin, int source_bin, double *covariance, double *kdd){
  
  (*covariance) = 0.0;
  (*kdd) = 0.0;

  for(int i = 0; i < this->w_values.size(); i++){
      (*covariance) += this->dw_values[i]*this->weight_values[i]*this->weight_values_kappa[source_bin][i]*this->G_kappa_coeffs_at_w[i][theta_bin][1];
      (*kdd) += this->dw_values[i]*this->weight_values[i]*this->weight_values[i]*this->weight_values_kappa[source_bin][i]*2.0*this->G_kappa_coeffs_at_w[i][theta_bin][2];
  }
  
}




void projection_stats::return_moments(int theta_bin, int source_bin, double delta, double source_bias, double *covariance, double *kdd){
  
  double w_1, w_2, eta, pw_1, pw_2;
  double cov_with_slice = 0.0;
  double w_theta = 0.0;
  double triple_with_slice = 0.0;
  double normalization = 1.0;
  double Omega_m = this->cosmo.Omega_m;
  
  if(theta_bin == 0){
    for(int i = 0; i < w_values.size(); i++){
        cov_with_slice = this->weight_values[i]*G_kappa_coeffs_at_w[i][theta_bin][1];
        triple_with_slice = this->weight_values[i]*this->weight_values[i]*2.0*G_kappa_coeffs_at_w[i][theta_bin][2];
        lognormal_deltaS_given_deltaL[i] = source_bias*expectation_of_kappa_given_delta_as_function_of_kdd_faster(delta, triple_with_slice, cov_with_slice);
    }
  }
  else{
    double delta_R1;
    double delta_R2;
    double R1_squared = this->bins[theta_bin-1]*this->bins[theta_bin-1];
    double R2_squared = this->bins[theta_bin]*this->bins[theta_bin];
    double one_over_area = 1.0/(R2_squared - R1_squared);

    for(int i = 0; i < w_values.size(); i++){
        cov_with_slice = this->weight_values[i]*G_kappa_coeffs_at_w[i][theta_bin][1];
        triple_with_slice = this->weight_values[i]*this->weight_values[i]*2.0*G_kappa_coeffs_at_w[i][theta_bin][2];
        delta_R2 = source_bias*expectation_of_kappa_given_delta_as_function_of_kdd_faster(delta, triple_with_slice, cov_with_slice);
        cov_with_slice = this->weight_values[i]*G_kappa_coeffs_at_w[i][theta_bin-1][1];
        triple_with_slice = this->weight_values[i]*this->weight_values[i]*2.0*G_kappa_coeffs_at_w[i][theta_bin-1][2];
        delta_R1 = source_bias*expectation_of_kappa_given_delta_as_function_of_kdd_faster(delta, triple_with_slice, cov_with_slice);
        lognormal_deltaS_given_deltaL[i] = delta_R2*R2_squared - delta_R1*R1_squared;
        lognormal_deltaS_given_deltaL[i] *= one_over_area;

    }
  }
  
  for(int i = 0; i < this->w_values.size(); i++){
    pw_1 = weight_values_source_density[source_bin][i];
    w_theta = lognormal_deltaS_given_deltaL[i];
    if(w_theta > -1.0) 
      normalization += this->dw_values[i]*pw_1*w_theta;
    else
      normalization -= this->dw_values[i]*pw_1;
  }
  
  for(int j = 0; j < this->w_values.size(); j++){
    int i = j;
    (*covariance) += this->source_bias_simplifications_cov[source_bin][theta_bin][j]*lognormal_deltaS_given_deltaL[j];
    (*kdd) += this->source_bias_simplifications_triple[source_bin][theta_bin][j]*lognormal_deltaS_given_deltaL[j];
  }
  
  (*covariance) /= normalization;
  (*kdd) /= normalization;
  
}




void projection_stats::set_source_bias_simplifications(){
  
  double w_1, w_2, eta, pw_1, pw_2;
  double Omega_m = this->cosmo.Omega_m;
  
  int n_kappa = this->kd_values.size();
  int n_w = this->w_values.size();
  int n_bin = this->G_kappa_coeffs_at_w[0].size();
  
  this->source_bias_simplifications = vector<vector<vector<double> > >(n_kappa, vector<vector<double> >(n_w, vector<double>(n_w, 0.0)));
  this->source_bias_simplifications_cov = vector<vector<vector<double> > >(n_kappa, vector<vector<double> >(n_bin, vector<double>(n_w, 0.0)));
  this->source_bias_simplifications_triple = vector<vector<vector<double> > >(n_kappa, vector<vector<double> >(n_bin, vector<double>(n_w, 0.0)));
  
  for(int l = 0; l < n_kappa; l++){
    for(int i = 0; i < this->w_values.size(); i++){
      w_1 = this->w_values[i];
      for(int j = i; j < this->w_values.size(); j++){
        w_2 = this->w_values[j];
        pw_2 = weight_values_source_density[l][j];
        this->source_bias_simplifications[l][i][j] = 1.5*Omega_m*w_1*this->one_over_a_values[i]*this->dw_values[j]*(w_2 - w_1)*this->one_over_w_values[j]*pw_2;
      }
    }
  }
  
  
  
  for(int l = 0; l < n_kappa; l++){
    for(int b = 0; b < n_bin; b++){
      for(int j = 0; j < this->w_values.size(); j++){
        for(int i = 0; i <= j; i++){
          this->source_bias_simplifications_cov[l][b][j] += this->dw_values[i]*this->weight_values[i]*this->G_kappa_coeffs_at_w[i][b][1]*this->source_bias_simplifications[l][i][j];
          this->source_bias_simplifications_triple[l][b][j] += this->dw_values[i]*this->weight_values[i]*this->weight_values[i]*2.0*this->G_kappa_coeffs_at_w[i][b][2]*this->source_bias_simplifications[l][i][j];
        }
      }
    }
  }
  
  
}





double projection_stats::expectation_of_kappa_given_delta_as_function_of_kdd_faster(double d, double kdd, double covariance){
  
  // If moments can be matched with a joint log-normal PDF
  if(kdd - 2.0*this->variance_of_delta*covariance*this->one_over_d0 > 0.0){
    double k0 = covariance*covariance*(1.0 + this->variance_of_delta*this->one_over_d0*this->one_over_d0)/(kdd - 2.0*covariance*this->variance_of_delta*this->one_over_d0);
    double COV = log(1.0 + covariance*this->one_over_d0/k0);
    if(d <= -d0) return -k0;
    return k0*(exp(0.5*COV*this->one_over_Gaussian_variance*(2.0*log(1.0+d*this->one_over_d0)+this->Gaussian_variance-COV))-1.0);
  }
  
  //otherwise just assume Gaussian PDF for kappa
  if(d <= -d0) return d*covariance/this->variance_of_delta;
  return covariance/(d0*this->Gaussian_variance)*(log(1.0+d/d0)+this->Gaussian_variance/2.0);
}



void Matter_2D::compute_PDF_and_mean_kappas_from_variable_kappa0_Bernardeau(double theta, vector<double> bins, vector<double> *delta_values, vector<double> *PDF, vector<vector<double> > *mean_deltas, vector<vector<vector<double> > > *mean_kappas, double* trough_variance, double* trough_skewness){

  
  
  int n_w = 150;
  int n_kappa = this->source_kernels.size();
  int n_bin = bins.size();
  int n;
  int n_source_bins = this->source_kernels.size();
  
  double delta;
  double D_delta;
  double delta_min;
  double delta_max;
  double kd, kdd, kkd, var_kappa_at_trough_radius;
  

  // For computing polynomial coefficients:
  vector<double> coefficients_phi;
  vector<vector<double> > coefficients_G_delta;
  vector<vector<vector<double> > > coefficients_G_kappa;
  
  vector<double> lambda_values(0, 0.0);
  vector<double> phi_lambda_values(0, 0.0);
  vector<double> phi_prime_values(0, 0.0);
  vector<vector<double> > G_delta_values(0, vector<double>(0, 0.0));
  vector<vector<vector<double> > > G_kappa_values(n_kappa, vector<vector<double> >(0, vector<double>(0, 0.0)));

  
  //cout << "Computing projected generating function...\n";
  this->matter->set_permanent_Legendres(constants::ell_max, theta, bins);

  compute_projected_phi_and_G_kappas_Bernardeau(n_w, theta, bins, &coefficients_phi, &coefficients_G_delta, &coefficients_G_kappa, &lambda_values, &phi_lambda_values, &phi_prime_values, &G_delta_values, &G_kappa_values, &kd, &kdd, &kkd, &var_kappa_at_trough_radius);
  vector<double> coefficients_G(coefficients_phi.size(), 0.0);
  vector<double> coefficients_G_prime(coefficients_phi.size(), 0.0);
  for(int i = 0; i < coefficients_phi.size()-1; i++) coefficients_G[i] = coefficients_phi[i+1]*double(i+1);
  for(int i = 0; i < coefficients_phi.size()-1; i++) coefficients_G_prime[i] = coefficients_G[i+1]*double(i+1);
  
  double variance = 2.0*coefficients_phi[2];
  double sigma = sqrt(variance);
  double skewness = 6.0*coefficients_phi[3];
  double delta_0 = get_delta0(variance, skewness);
  double sigma_Gauss = sqrt(log(1.0 + variance/delta_0/delta_0));
  double mean_Gauss = log(delta_0) - 0.5*sigma_Gauss*sigma_Gauss;
  (*trough_variance) = variance;
  (*trough_skewness) = skewness;
  
  double covariance = kd;
  double triple = kkd;
  double k0_at_trough_radius = get_kappa0(delta_0, variance, covariance, triple);
  double var_kappa_at_trough_radius_lognormal = get_var_kappa(delta_0, k0_at_trough_radius, covariance, kkd);
  
  D_delta = 0.01*sigma;
  delta_min = -delta_0;
  delta_max = exp(mean_Gauss + 5.0*sigma_Gauss) - delta_0;
  delta_max = max(delta_max, 2.0);
  int index = find_index(600.0*constants::arcmin, &bins);
  covariance = coefficients_G_kappa[0][index][1];
  triple = 2.0*coefficients_G_kappa[0][index][2];
  
  cout << "        delta_0: " << delta_0 << '\n';
  cout << "        kappa_0: " << k0_at_trough_radius << '\n';
  cout << "      delta_max: " << delta_max << '\n';
  cout << "        sigma_T: " << sigma << '\n';
  cout << "          Var_T: " << variance << '\n';
  cout << "         Skew_T: " << skewness << '\n';
  cout << "     covariance: " << covariance << '\n';
  cout << "<kappa delta^2>: " << triple << '\n';/**/
  n = 1 + int((delta_max - delta_min)/D_delta + 0.1);
  
  delta_values->resize(n, 0.0);
  PDF->resize(n, 0.0);
  mean_deltas->resize(n_bin, vector<double>(n, 0.0));
  mean_kappas->resize(n_source_bins, vector<vector<double> >(n_bin, vector<double>(n, 0.0)));
  
  cout << "Computing projected generating function...\n";
  //compute_projected_phi(n_w, theta, delta_min, &lambda_values, &phi_lambda_values, &phi_prime_values);
  int n_lambda = lambda_values.size();
  int n_tau = 30;
  double dl = lambda_values[1] - lambda_values[0];
  vector<double> tau_of_lambda(n_lambda, 0.0);
  
  cout << "Computing polynomial coefficients...\n";
  for(int i = 0; i < n_lambda; i++){
    if(abs(lambda_values[i]) < 0.1*dl)
      tau_of_lambda[i] = 0.0;
    else
      tau_of_lambda[i] = sqrt(2.0*(phi_prime_values[i]*lambda_values[i] - phi_lambda_values[i]));
    
    if(lambda_values[i] < 0.0) tau_of_lambda[i] *= -1;
    if(i > 0 && tau_of_lambda[i] < tau_of_lambda[i-1]) tau_of_lambda[i] = tau_of_lambda[i-1];
  }
  double tau_max = tau_of_lambda[n_lambda - 1];
  double tau_min = tau_of_lambda[0];
  double dtau;
  vector<double> tau_values(n_tau-1, 0.0);
  vector<double> y_values(n_tau-1, 0.0);
  vector<double> G_eff_values(n_tau-1, 0.0);
  vector<double> G_eff_prime_values(n_tau-1, 0.0);
  vector<vector<double> > G_delta_tau_values(bins.size(), vector<double>(n_tau-1, 0.0));
  vector<vector<vector<double> > > G_kappa_tau_values(n_kappa, vector<vector<double> >(bins.size(), vector<double>(n_tau-1, 0.0)));
      
  dtau = -tau_min/double(n_tau/2);
  for(int i = 0; i < n_tau/2; i++){
    tau_values[i] = tau_min + double(i)*dtau;
  }
  tau_values[n_tau/2] = 0.0;
  dtau = tau_max/double(n_tau - n_tau/2 - 1);
  for(int i = n_tau/2 + 1 ; i < n_tau-1; i++){
    tau_values[i] = double(i - n_tau/2)*dtau;
  }
  
  
  for(int i = 0; i < n_tau-1; i++){
    y_values[i] = interpolate_Newton(tau_values[i], &tau_of_lambda, &lambda_values, this->order);
    G_eff_values[i] = interpolate_Newton(tau_values[i], &tau_of_lambda, &phi_prime_values, this->order);
    G_eff_prime_values[i] = tau_values[i]/(interpolate_Newton(tau_values[i], &tau_of_lambda, &lambda_values, this->order));
    for(int b = 0; b < bins.size(); b++){
      G_delta_tau_values[b][i] = interpolate_Newton(tau_values[i], &tau_of_lambda, &G_delta_values[b], this->order);
    }
    for(int s = 0; s < n_kappa; s++){
      for(int b = 0; b < bins.size(); b++){
        G_kappa_tau_values[s][b][i] = interpolate_Newton(tau_values[i], &tau_of_lambda, &G_kappa_values[s][b], this->order);
      }
    }
  }
  
  
  
  double dr;
  double lambda_c;
  double tau_c;
  complex<double> step;
  complex<double> first_step;
  complex<double> lambda;
  complex<double> tau;
  complex<double> dlambda;
  complex<double> G;
  complex<double> G_delta_complex;
  complex<double> G_kappa_complex;
  complex<double> exponent;
  complex<double> lambda_next;
  complex<double> tau_next;
  complex<double> G_next;
  complex<double> G_delta_complex_next;
  complex<double> G_delta_step;
  complex<double> G_kappa_complex_next;
  complex<double> G_kappa_step;
  complex<double> exponent_next;
  
  vector<double> fact = factoria(coefficients_phi.size());
    
    vector<double> coefficients_tau;
    coefficients_tau = get_tau_coefficients(&coefficients_phi);
    vector<double> coefficients_G_tau_aux = return_coefficients(&tau_values, &G_eff_values, coefficients_phi.size()-1);
    vector<double> coefficients_tau_prime(coefficients_tau.size(), 0.0);
    vector<double> coefficients_G_of_tau(coefficients_tau.size(), 0.0);
    vector<double> coefficients_G_prime_of_tau(coefficients_tau.size(), 0.0);
    vector<double> coefficients_G_prime_prime_of_tau(coefficients_tau.size(), 0.0);
    vector<vector<double> > coefficients_G_delta_of_tau(bins.size(), vector<double>(coefficients_tau.size(), 0.0));
    vector<vector<vector<double> > > coefficients_G_kappa_of_tau(n_kappa, vector<vector<double> >(bins.size(), vector<double>(coefficients_tau.size(), 0.0)));

    vector<vector<double> > Bell_matrix(0, vector<double>(0, 0.0));
    vector<vector<double> > inverse_Bell_matrix(0, vector<double>(0, 0.0));
    return_Bell_matrix(&Bell_matrix, &inverse_Bell_matrix, &coefficients_tau);
    
    for(int i = 0; i < coefficients_tau.size(); i++){
      for(int j = 0; j <= i; j++){
        coefficients_G_of_tau[i] += inverse_Bell_matrix[i][j]*coefficients_G[j]*fact[j];
      }
      coefficients_G_of_tau[i] /= fact[i];
    }
    for(int i = 3; i < coefficients_tau.size(); i++){
      coefficients_G_of_tau[i] = coefficients_G_tau_aux[i];
    }
    
    for(int i = 0; i < coefficients_tau.size()-1; i++){
      coefficients_G_prime_of_tau[i] = coefficients_G_of_tau[i+1]*double(i+1);
      coefficients_tau_prime[i] = coefficients_tau[i+1]*double(i+1);
    }
    for(int i = 0; i < coefficients_tau.size()-1; i++){
      coefficients_G_prime_prime_of_tau[i] = coefficients_G_prime_of_tau[i+1]*double(i+1);
    }
    
    
    
    
  cout << "Coefficients in new method:\n";
  for(int i = 0; i < coefficients_tau.size(); i++){
      cout << i << setw(20);
      cout << coefficients_G_of_tau[i] << setw(20);
      cout << coefficients_G_prime_of_tau[i] << '\n';
  }
  
  for(int b = 0; b < bins.size(); b++){
    coefficients_G_tau_aux = return_coefficients(&tau_values, &G_delta_tau_values[b], coefficients_phi.size()-1);

    for(int i = 0; i < coefficients_tau.size(); i++){
      for(int j = 0; j <= i; j++){
        coefficients_G_delta_of_tau[b][i] += inverse_Bell_matrix[i][j]*coefficients_G_delta[b][j]*fact[j];
      }
      coefficients_G_delta_of_tau[b][i] /= fact[i];
    }
    for(int i = 3; i < coefficients_tau.size(); i++){
      coefficients_G_delta_of_tau[b][i] = coefficients_G_tau_aux[i];
    }
  }
  
  for(int s = 0; s < n_kappa; s++){
    for(int b = 0; b < bins.size(); b++){
      coefficients_G_tau_aux = return_coefficients(&tau_values, &G_kappa_tau_values[s][b], coefficients_phi.size()-1);
    
      for(int i = 0; i < coefficients_tau.size(); i++){
        for(int j = 0; j <= i; j++){
          coefficients_G_kappa_of_tau[s][b][i] += inverse_Bell_matrix[i][j]*coefficients_G_kappa[s][b][j]*fact[j];
        }
        coefficients_G_kappa_of_tau[s][b][i] /= fact[i];
      }
      for(int i = 3; i < coefficients_tau.size(); i++){
        coefficients_G_kappa_of_tau[s][b][i] = coefficients_G_tau_aux[i];
      }
    }
  }

    
    cout << "Computing PDF...\n";
    for(int i = 0; i < n; i++){
      delta = delta_min + double(i)*D_delta;
      (*delta_values)[i] = delta;
      (*PDF)[i] = 0.0;
      
      if(delta < phi_prime_values[n_lambda - 1]){
        tau_c = interpolate_Newton(delta, &phi_prime_values, &tau_of_lambda, this->order);
        lambda_c = interpolate_Newton(delta, &phi_prime_values, &lambda_values, this->order);
      }
      else{
        tau_c = tau_of_lambda[n_lambda - 1];
        lambda_c = lambda_values[n_lambda - 1];
      }
      
      lambda = complex<double>(lambda_c, 0.0);
      tau = complex<double>(tau_c, 0.0);
      G = complex<double>(delta, 0.0);
      exponent = complex<double>(exp(-0.5*pow(tau_c, 2)), 0.0);
      
      // sigma_r^2 \approx 1/phi''(lambda_c)
      dr = 0.01/sqrt(interpolate_neville_aitken_derivative(lambda_c, &lambda_values, &phi_prime_values, this->order));
      dlambda = complex<double>(0.0, dr);
      int j = 0;
      do{
        lambda_next = lambda + 0.5*dlambda;
        tau_next = get_tau_from_secant_method_complex_Bernardeau_notation_2D(lambda_next, tau, &coefficients_G_prime_of_tau, &coefficients_G_prime_prime_of_tau);
        G_next = return_polnomial_value(tau_next, &coefficients_G_of_tau);
        dlambda = -dr*conj(G_next-delta)/abs(G_next-delta);
        lambda_next = lambda + dlambda;
        tau_next = get_tau_from_secant_method_complex_Bernardeau_notation_2D(lambda_next, tau_next, &coefficients_G_prime_of_tau, &coefficients_G_prime_prime_of_tau);
        G_next = return_polnomial_value(tau_next, &coefficients_G_of_tau);
        exponent_next = exp(lambda_next*(G_next-delta)-0.5*pow(tau_next, 2));
        
        step = 0.5*dlambda*(exponent_next+exponent);
        (*PDF)[i] += step.imag();
        
        for(int b = 0; b < bins.size(); b++){
          G_delta_complex = return_polnomial_value(tau, &coefficients_G_delta_of_tau[b]);
          G_delta_complex_next = return_polnomial_value(tau_next, &coefficients_G_delta_of_tau[b]);
          G_delta_step = 0.5*dlambda*(exponent_next*G_delta_complex_next+exponent*G_delta_complex);
          (*mean_deltas)[b][i] += G_delta_step.imag();
        }
          
        for(int s = 0; s < n_kappa; s++){
          for(int b = 0; b < bins.size(); b++){
            G_kappa_complex = return_polnomial_value(tau, &coefficients_G_kappa_of_tau[s][b]);
            G_kappa_complex_next = return_polnomial_value(tau_next, &coefficients_G_kappa_of_tau[s][b]);
            G_kappa_step = 0.5*dlambda*(exponent_next*G_kappa_complex_next+exponent*G_kappa_complex);
            (*mean_kappas)[s][b][i] += G_kappa_step.imag();
          }
        }

        lambda = lambda_next;
        dlambda = -dr*conj(G_next-delta)/abs(G_next-delta);
        tau = tau_next;
        G = G_next;
        exponent = exponent_next;
        if(j == 0) first_step = step;
        j++;
      }while(abs(step/first_step) > 1.0e-4 || j < 600);

      (*PDF)[i] /= constants::pi;
      
      for(int b = 0; b < bins.size(); b++)
        (*mean_deltas)[b][i] /= constants::pi*(*PDF)[i];
      
      for(int s = 0; s < n_kappa; s++)
        for(int b = 0; b < bins.size(); b++)
          (*mean_kappas)[s][b][i] /= constants::pi*(*PDF)[i];
    
      
      if(i%1 == 0){
        cout << delta << setw(20);
        cout << (*PDF)[i] << setw(20);
        cout << PDF_of_delta(delta, delta_0, variance) << setw(20);
        cout << exp(-0.5*delta*delta/variance)/sqrt(2.0*constants::pi*variance) << '\n';
      }
      
    }
    
    this->compute_moments(delta_values, PDF);
    
  
}









void Matter_2D::compute_moments(double theta, double* trough_variance, double* trough_skewness){
  
    int n_w = 20;

    set_2D_stats(n_w, theta);


    (*trough_variance) = this->stats_2D.variance_of_delta;
    (*trough_skewness) = this->stats_2D.skewness_of_delta;
}

//Marco was here

void Matter_2D::compute_moments(double* theta, int size_theta, vector<vector<double>>* trough_variance, vector<vector<double>>* trough_skewness,long* bins2, int len_bins2,long* bins3, int len_bins3){

    int n_w = 20;


    set_2D_stats(n_w, size_theta, theta,  bins2, len_bins2,bins3,len_bins3); //Marco was here


    (*trough_variance) =  this->stats_2D.variance_array2;
    (*trough_skewness) =  this->stats_2D.skewness_array2;

}


//Marco was also heeeere! generalization to nbins
void Matter_2D::set_2D_stats(int n_w, int size_theta, double* theta,long* bins2, int len_bins2,long* bins3, int len_bins3){


  clock_t start = clock();
  clock_t start1 = clock();
  int n_kappa = this->lens_kernels.size();




  this->stats_2D.variance_array2 = vector<vector<double>>(size_theta, vector<double> (len_bins2, 0.0));
  this->stats_2D.skewness_array2 = vector<vector<double>>(size_theta, vector<double> (len_bins3, 0.0));
  this->matter->set_permanent_Legendres_array(constants::ell_max, theta, size_theta);
  this->matter->set_permanent_Legendres(constants::ell_max, theta[0]);
  /*
  if(theta == 0.0){
    //cout<<"smoothing scale 0.0"<<endl;

    start = clock();
    //routine in
    this->matter->set_permanent_Legendres_null(constants::ell_max, 0.000);
    std::cout << "setting permanent leg : " << clock() - start << "ms \n";
  }else{

  }
  */


  this->stats_2D.uni = this->universe;
  this->stats_2D.cosmo = this->cosmology;

  //int nq = this->stats_2D.Cell_delta.size();
  //this->stats_2D.Cell_delta = vector<double> (constants::ell_max, 0.0); //Marco here
  //this->stats_2D.Cells_kappa_total = vector<vector<double> >((len_bins2, vector<double>(nq, 0.0)); //Marco here


  int order = 3;

  double dw = (this->w_max - this->w_min)/double(n_w);
  vector<double> R_arr(size_theta, 0.0);
  double eta;
  double weight;
  double weight1;
  double weight2;

  this->stats_2D.w_values = vector<double>(0, 0.0);
  this->stats_2D.dw_values = vector<double>(0, 0.0);


  vector<double> w_boundaries(0, 0.0);
  vector<double> p_of_w(0, 0.0);


  if(this->lens_kernels[0].histogram_modus == 1){

    int n = 0, steps_per_histogram_bin = 1;
    this->lens_kernels[0].return_w_boundaries_and_pofw(&w_boundaries, &p_of_w);
  

    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0) n++;
    }

    while(n*steps_per_histogram_bin < n_w){
      steps_per_histogram_bin++;
    }

    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0){
        dw = (w_boundaries[i+1] - w_boundaries[i])/double(steps_per_histogram_bin);
        double sum = 0.0;

        for(int j = 0; j < steps_per_histogram_bin; j++){
          sum += pow(w_boundaries[i]+dw*(0.5 + double(j)), 2.0);
        }

        sum /= double(steps_per_histogram_bin);


        for(int j = 0; j < steps_per_histogram_bin; j++){
          this->stats_2D.w_values.push_back(w_boundaries[i]+dw*(0.5 + double(j)));
          this->stats_2D.dw_values.push_back(dw);
        }
      }
    }
  }
  else{

    for(double w = this->w_min + 0.5*dw; w < this->w_max; w+= dw){
      this->stats_2D.w_values.push_back(w);
      this->stats_2D.dw_values.push_back(dw);
    }
  }

  this->stats_2D.weight_values = vector<double>(this->stats_2D.w_values.size(), 0.0);
  this->stats_2D.weight_values_2 = vector<double>(this->stats_2D.w_values.size(), 0.0);

  this->stats_2D.G_coeffs_at_w = vector<vector<double> >(this->stats_2D.w_values.size(), vector<double>(order+1, 0.0));
  this->stats_2D.G_coeffs_at_w_arr = vector<vector<vector<double> > >(this->stats_2D.w_values.size(), vector<vector<double>>(size_theta, vector<double>(order+1, 0.0)));


  this->stats_2D.variance_of_delta = 0.0;
  this->stats_2D.skewness_of_delta = 0.0;



  for(int jj = 0; jj < len_bins2; jj++){
        if(this->sd_lensing_A ==0 or this->sd_lensing_A ==1){
            this->lens_kernels[bins2[2*jj]-1].shear_or_delta = this->sd_lensing_A;

        }
        if(this->sd_lensing_B ==0 or this->sd_lensing_B ==1){
            this->lens_kernels[bins2[2*jj+1]-1].shear_or_delta = this->sd_lensing_B;
        }

    }

  for(int jj = 0; jj < len_bins3; jj++){
        if(this->sd_lensing_A ==0 or this->sd_lensing_A ==1){

            this->lens_kernels[bins3[3*jj]-1].shear_or_delta = this->sd_lensing_A;
        }
        if(this->sd_lensing_B ==0 or this->sd_lensing_B ==1){
            this->lens_kernels[bins3[3*jj+1]-1].shear_or_delta = this->sd_lensing_B;
        }
       }


  double mute;

  int countt = 0;
  double w_max = 0;
  for(int i = 0; i < this->stats_2D.w_values.size(); i++){
    countt+=1;
    double w = this->stats_2D.w_values[i];
    w_max = w;
    eta = this->universe->eta_at_a(1.0) - w;
    for( int th=0; th<size_theta; th++){
        R_arr[th] = w*c_over_e5*theta[th];
    }

    this->matter->compute_variance_and_skewness_arr(size_theta, R_arr, eta, &this->stats_2D.G_coeffs_at_w_arr[i]); // routine in matter_ouput.h






    // ***********************************************
    for(int jj = 0; jj < len_bins2; jj++){

        double D_growth = this->matter->growth(eta);
        double IA = (this->A0)*pow((this->lens_kernels[bins2[2*jj]-1].w_to_z(w)/1+this->z0),this->alpha0)*0.0134*this->cosmology.Omega_m/D_growth;
        double IA1 = (this->A0)*pow((this->lens_kernels[bins2[2*jj+1]-1].w_to_z(w)/1+this->z0),this->alpha0)*0.0134*this->cosmology.Omega_m/D_growth;

        weight = this->lens_kernels[bins2[2*jj]-1].weight_at_comoving_distance(w)*w -this->lens_kernels[bins2[2*jj]-1].nz_at_comoving_distance(w)*IA*w ;
        weight1 = this->lens_kernels[bins2[2*jj+1]-1].weight_at_comoving_distance(w)*w -this->lens_kernels[bins2[2*jj+1]-1].nz_at_comoving_distance(w)*IA1*w;

        mute = weight*weight1;
        //cout<<mute<<"  "<< this->lens_kernels[bins2[2*jj]-1].weight_at_comoving_distance(w)*w* this->lens_kernels[bins2[2*jj+1]-1].weight_at_comoving_distance(w)*w<<endl;
        //double mute1 = this->lens_kernels[bins2[2*jj]-1].nz_at_comoving_distance(w)*IA*w ;
        //double mute2 = this->lens_kernels[bins2[2*jj]-1].weight_at_comoving_distance(w)*w ;
        //cout <<1./(this->lens_kernels[bins2[2*jj]-1].weight_at_comoving_distance(w)/(this->lens_kernels[bins2[2*jj]-1].nz_at_comoving_distance(w)*IA))<<"  "<<   IA  <<"  "<<(this->lens_kernels[bins2[2*jj]-1].w_to_z(w))<<"   "<<this->lens_kernels[bins2[2*jj]-1].weight_at_comoving_distance(w)<<"  "<<this->lens_kernels[bins2[2*jj]-1].nz_at_comoving_distance(w)<< "  "<< mute2 << " " <<mute1<<endl;

        for (int k=0; k<size_theta; k++){

            this->stats_2D.variance_array2[k][jj] +=  this->stats_2D.dw_values[i]*mute*this->stats_2D.G_coeffs_at_w_arr[i][k][2]*2.0;
        }
    }


    for(int jj = 0; jj < len_bins3; jj++){
        double D_growth = this->matter->growth(eta);
        double IA = (this->A0)*pow((this->lens_kernels[bins3[3*jj]-1].w_to_z(w)/1+this->z0),this->alpha0)*0.0134*this->cosmology.Omega_m/D_growth;
        double IA1 = (this->A0)*pow((this->lens_kernels[bins3[3*jj+1]-1].w_to_z(w)/1+this->z0),this->alpha0)*0.0134*this->cosmology.Omega_m/D_growth;
        double IA2 = (this->A0)*pow((this->lens_kernels[bins3[3*jj+2]-1].w_to_z(w)/1+this->z0),this->alpha0)*0.0134*this->cosmology.Omega_m/D_growth;

        weight = this->lens_kernels[bins3[3*jj]-1].weight_at_comoving_distance(w)*w -this->lens_kernels[bins3[3*jj]-1].nz_at_comoving_distance(w)*IA*w ;
        weight1 = this->lens_kernels[bins3[3*jj+1]-1].weight_at_comoving_distance(w)*w -this->lens_kernels[bins3[3*jj+1]-1].nz_at_comoving_distance(w)*IA1*w;
        weight2 = this->lens_kernels[bins3[3*jj+2]-1].weight_at_comoving_distance(w)*w -this->lens_kernels[bins3[3*jj+2]-1].nz_at_comoving_distance(w)*IA2*w;



        mute = weight*weight1*weight2;
        for (int k=0; k<size_theta; k++){

            this->stats_2D.skewness_array2[k][jj] +=  this->stats_2D.dw_values[i]*mute*this->stats_2D.G_coeffs_at_w_arr[i][k][3]*6.0;
        }
    }


  }


    // save_results_delta_2_3 **********************************

    // save_results_delta_2_3 **********************************
    remove("z_.txt");
    FILE* F0 = fopen("z_.txt", "w");
    fclose(F0);
    fstream outqq,outqq1;
    outqq.open("z_.txt");
    for(int i = 0; i < this->stats_2D.w_values.size(); i++){

        double w = this->stats_2D.w_values[i];
        //cout<< this->stats_2D.dw_values[i]<<" "<<this->lens_kernels[bins2[0]-1].weight_at_comoving_distance(w)<<"  "<<w<<"  "<<this->lens_kernels[bins2[0]-1].w_to_z(w)<<endl;
        eta = this->universe->eta_at_a(1.0) - w;
        double D_growth = this->matter->growth(eta);

        //cout<<this->lens_kernels[bins2[0]-1].nz_at_comoving_distance(w)<<"    "<<D_growth<<endl;

        outqq <<this->lens_kernels[0].w_to_z(w)<<endl;
    }
    outqq.close();
    for(int i = 0; i < this->stats_2D.w_values.size(); i++){

        double w = this->stats_2D.w_values[i];
        //outqq1<< this->stats_2D.dw_values[i]<<" "<<this->lens_kernels[bins2[0]-1].weight_at_comoving_distance(w)<<"  "<<this->lens_kernels[bins2[1]-1].weight_at_comoving_distance(w)<<"  "<<this->lens_kernels[bins2[2]-1].weight_at_comoving_distance(w)<<"  "<<this->lens_kernels[bins2[3]-1].weight_at_comoving_distance(w)<<"  "<<this->lens_kernels[bins2[4]-1].weight_at_comoving_distance(w)<<"  "<<w<<"  "<<this->lens_kernels[bins2[0]-1].w_to_z(w)<<endl;
        outqq1<< this->stats_2D.dw_values[i]<<" "<<this->lens_kernels[bins2[0]-1].weight_at_comoving_distance(w)<<"  "<<w<<"  "<<this->lens_kernels[bins2[0]-1].w_to_z(w)<<endl;

        eta = this->universe->eta_at_a(1.0) - w;
        double D_growth = this->matter->growth(eta);

        //cout<<this->lens_kernels[bins2[0]-1].nz_at_comoving_distance(w)<<"    "<<D_growth<<endl;

        //outqq <<this->lens_kernels[0].w_to_z(w)<<endl;
    }
    outqq1.close();

    cout<<"saving"<<endl;
    char* mute_c;
    this->matter->return_out_d23(0,&mute_c);
    remove(mute_c);
    FILE* F = fopen(mute_c, "w");
    fclose(F);
    fstream out;
    out.open(mute_c);
    out <<"# theta: "<<size_theta<<endl;
    out <<"# z: "<<this->stats_2D.w_values.size()<<endl;
    for(int i = 0; i < this->stats_2D.w_values.size(); i++){
        out << scientific << setprecision(5);
        for(int sm = 0; sm <size_theta; sm++){

            out << this->stats_2D.G_coeffs_at_w_arr[i][sm][2]<<setw(15);

            fflush(stdout);

      }
        out <<"\n";
    }
    out.close();


    this->matter->return_out_d23(1,&mute_c);
    cout<<mute_c<<endl;
    remove(mute_c);
    FILE* F1 = fopen(mute_c, "w");
    fclose(F1);
    fstream out1;
    out1.open(mute_c);
    out1 <<"# theta: "<<size_theta<<endl;
    out1 <<"# z: "<<this->stats_2D.w_values.size()<<endl;
    for(int i = 0; i < this->stats_2D.w_values.size(); i++){
        out1 << scientific << setprecision(5);
        for(int sm = 0; sm <size_theta; sm++){

            out1 << this->stats_2D.G_coeffs_at_w_arr[i][sm][3]<<setw(15);
            fflush(stdout);



      }
        out1 <<"\n";
    }
    out1.close();

}








void Matter_2D::set_2D_stats(int n_w, double theta){

  // generalize to a given number of lens_bins
  int n_kappa = this->lens_kernels.size();

  // this is in case one wants to change the lensing/delta kernel from imput
  for(int i = 0; i < n_kappa; i++){

        if(this->sd_lensing_A ==0){
            this->lens_kernels[i].shear_or_delta = this->sd_lensing_A;
            this->lens_kernels_2[i].shear_or_delta = this->sd_lensing_B;
        }
        if(this->sd_lensing_A ==1){
            this->lens_kernels[i].shear_or_delta = this->sd_lensing_A;
            this->lens_kernels_2[i].shear_or_delta = this->sd_lensing_B;
        }

  }
  int nq = this->stats_2D.Cell_delta.size();

  this->matter->set_permanent_Legendres(constants::ell_max, theta);
  this->stats_2D.uni = this->universe;
  this->stats_2D.cosmo = this->cosmology;
  this->stats_2D.Cell_delta = vector<double> (constants::ell_max, 0.0); //Marco here
  this->stats_2D.Cells_kappa_total = vector<vector<double> >((n_kappa*(n_kappa+1))/2, vector<double>(nq, 0.0)); //Marco here


  int order = 3;

  double dw = (this->w_max - this->w_min)/double(n_w);
  double R;
  double eta;
  double weight;
  double weight1;

  // IN BERNARDEAU NOTATION, I.E. LAMBDA INSTEAD OF Y, AND PHI BEING THE CUMULANT GENERATING FUNCTION:
  this->stats_2D.w_values = vector<double>(0, 0.0);
  this->stats_2D.dw_values = vector<double>(0, 0.0);
  
  vector<double> w_boundaries(0, 0.0);
  vector<double> p_of_w(0, 0.0);
  
  if(this->lens_kernels[0].histogram_modus == 1){
    int n = 0, steps_per_histogram_bin = 1;
    this->lens_kernels[0].return_w_boundaries_and_pofw(&w_boundaries, &p_of_w);


    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0) n++;
    }
    while(n*steps_per_histogram_bin < n_w){
      steps_per_histogram_bin++;
    }
    for(int i = 0; i < p_of_w.size(); i++){
      if(p_of_w[i] != 0.0){
        dw = (w_boundaries[i+1] - w_boundaries[i])/double(steps_per_histogram_bin);
        double sum = 0.0;
        
        for(int j = 0; j < steps_per_histogram_bin; j++){
          sum += pow(w_boundaries[i]+dw*(0.5 + double(j)), 2.0);
        }
        
        sum /= double(steps_per_histogram_bin);
        
        for(int j = 0; j < steps_per_histogram_bin; j++){
          this->stats_2D.w_values.push_back(w_boundaries[i]+dw*(0.5 + double(j)));
          this->stats_2D.dw_values.push_back(dw);
        }
      }
    }
  }
  else{
    for(double w = this->w_min + 0.5*dw; w < this->w_max; w+= dw){
      this->stats_2D.w_values.push_back(w);
      this->stats_2D.dw_values.push_back(dw);
    }
  }


  this->stats_2D.weight_values = vector<double>(this->stats_2D.w_values.size(), 0.0);
  this->stats_2D.weight_values_2 = vector<double>(this->stats_2D.w_values.size(), 0.0);

  this->stats_2D.G_coeffs_at_w = vector<vector<double> >(this->stats_2D.w_values.size(), vector<double>(order+1, 0.0));
      
  this->stats_2D.variance_of_delta = 0.0;
  this->stats_2D.skewness_of_delta = 0.0;


  //MODIFY THIS TO SET R = 0
  if (this->Cl_compute == 1){
        this->matter->set_permanent_Legendres_null(constants::ell_max, 0.000);
  }
  for(int i = 0; i < this->stats_2D.w_values.size(); i++){
    
    double w = this->stats_2D.w_values[i];
    
    eta = this->universe->eta_at_a(1.0) - w;
    R = w*c_over_e5*theta;

    
    weight = this->lens_kernels[0].weight_at_comoving_distance(w)*w;
    weight1 = this->lens_kernels_2[0].weight_at_comoving_distance(w)*w;  //Marco was here


    this->stats_2D.weight_values[i] = weight*weight1; //Marco was here
    this->stats_2D.weight_values_2[i] = weight*weight1*weight; //Marco was here


    this->matter->compute_variance_and_skewness(R, eta, &this->stats_2D.G_coeffs_at_w[i]); // routine in matter_ouput.h



    //Marco was here (changed pow of the kernels)
    this->stats_2D.variance_of_delta += this->stats_2D.dw_values[i]*stats_2D.weight_values[i]*this->stats_2D.G_coeffs_at_w[i][2]*2.0;

    this->stats_2D.skewness_of_delta += this->stats_2D.dw_values[i]*stats_2D.weight_values_2[i]*this->stats_2D.G_coeffs_at_w[i][3]*6.0;

  }
}


