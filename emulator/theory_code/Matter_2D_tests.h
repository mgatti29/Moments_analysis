


void Matter_2D::test_wtheta(int n_w, int source_bin, double theta, double delta, vector<double> bins){

  int order = 3;
  int n_kappa = this->source_kernels.size();
  
  double dw = (this->w_max - this->w_min)/double(n_w);
  double R;
  double eta;
  double weight;
  
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
    
  
  
  double var = 0.0;
  double skew = 0.0;
  
  for(int i = 0; i < w_values.size(); i++){
    
    double w = w_values[i];
    cout << "\rProjecting 3D statistics along co-moving distance: " << w << "/" << w_values[w_values.size()-1] << "     ";
    cout.flush();
    
    eta = this->universe->eta_at_a(1.0) - w;
    R = w*c_over_e5*theta;
    
    for(int b = 0; b < physical_bins.size(); b++){
      physical_bins[b] = bins[b]*w*c_over_e5;
    }

    this->matter->compute_lognormal_final(R, eta, &physical_bins, &G_coeffs_at_w[i], &G_kappa_coeffs_at_w[i]);

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
    
    var += dw_values[i]*pow(weight_values[i], 2)*G_coeffs_at_w[i][2]*2.0;
    skew += dw_values[i]*pow(weight_values[i], 3)*G_coeffs_at_w[i][3]*6.0;
  }

  double d0 = get_delta0(var, skew);
  double cov_with_slice = 0.0;
  double triple_with_slice = 0.0;
  
  int index = find_index(theta, &bins);
  
  vector<double> Gaussian_deltaS_given_deltaL(w_values.size(), 0.0);
  vector<double> lognormal_deltaS_given_deltaL(w_values.size(), 0.0);
  
  for(int i = 0; i < w_values.size(); i++){
      cov_with_slice = weight_values[i]*2.0*G_coeffs_at_w[i][2];
      triple_with_slice = weight_values[i]*weight_values[i]*6.0*G_coeffs_at_w[i][3];
      Gaussian_deltaS_given_deltaL[i] = delta*cov_with_slice/var;
      lognormal_deltaS_given_deltaL[i] = expectation_of_kappa_given_delta_as_function_of_kdd(delta, d0, triple_with_slice, var, cov_with_slice);
  }
  
  double w_theta_2D = expectation_of_kappa_given_delta_as_function_of_kdd(delta, d0, kdd_values_source_density[source_bin], var, kd_values_source_density[source_bin]);
  double w_theta_projected = 0.0;
  cout << kdd_values_source_density[source_bin] - 2.0*kd_values_source_density[source_bin]*var/d0 << '\n';
  
  for(int i = 0; i < w_values.size(); i++){
      //w_theta_projected += dw_values[i]*weight_values_source_density[source_bin][i]*Gaussian_deltaS_given_deltaL[i];
      w_theta_projected += dw_values[i]*weight_values_source_density[source_bin][i]*lognormal_deltaS_given_deltaL[i];
  }
  
  cout << "comparing projected w(theta) values at theta_T = " << theta/constants::arcmin << " :\n";
  cout << "               d0 = " << d0 << '\n';
  cout << "            delta = " << delta << '\n';
  cout << "       w_theta_2D = " << w_theta_2D << '\n';
  cout << "w_theta_projected = " << w_theta_projected << '\n';

  
}



