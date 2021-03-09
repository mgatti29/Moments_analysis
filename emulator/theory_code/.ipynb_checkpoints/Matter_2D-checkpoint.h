

class projection_stats{
  
 public:
  projection_stats() : initialized(0) {}
  int initialized;
  
  double d0;
  double one_over_d0;
  double one_over_d0_squared;
  double variance_of_delta;
  double Gaussian_variance;
  double one_over_Gaussian_variance;
  double skewness_of_delta;
  
  vector<double> bins;
  vector<double> w_values;
  vector<double> one_over_w_values;
  vector<double> one_over_a_values;
  vector<double> dw_values;
  vector<double> weight_values;
  vector<double> weight_values_2; //Marco here
  vector<double> variance_array; //Marco here
  vector<double> skewness_array; //Marco here
  vector<vector<double>> variance_array2; //Marco here
  vector<vector<double>> skewness_array2; //Marco here

  vector<vector<double> > weight_values_kappa;
  vector<vector<double> > weight_values_source_density;
  
  vector<double> k0_values;
  vector<double> kd_values;
  vector<double> kdd_values;
  vector<double> kkd_values;
  vector<double> kk_values_total;
  vector<double> kk_values_correlated;
  vector<double> kk_values_uncorrelated;
  
  vector<double> Cell_delta;
  vector<vector<double> > Cells_cross;
  vector<vector<double> > Cells_kappa;
  vector<vector<double> > Cells_kappa_total; //Marco was here
  vector<vector<double> > Cells_kappa_uncorrelated;
    
  vector<vector<double> > G_coeffs_at_w;
  vector<vector<vector<double> > > G_coeffs_at_w_arr; //Marco was here
  vector<vector<vector<double> > > G_kappa_coeffs_at_w;
  
  vector<double> new_weights;
  vector<double> lognormal_deltaS_given_deltaL;
  
  Universe *uni;
  cosmological_model cosmo;
  
  vector<vector<vector<double> > > source_bias_simplifications;
  vector<vector<vector<double> > > source_bias_simplifications_cov;
  vector<vector<vector<double> > > source_bias_simplifications_triple;
  void set_source_bias_simplifications();
  void return_moments_delta(int theta_bin, double *covariance, double *kdd);
  void return_moments(int theta_bin, int source_bin, double *covariance, double *kdd);
  void return_moments(int theta_bin, int source_bin, double delta, double source_bias, double *covariance, double *kdd);

  double expectation_of_kappa_given_delta_as_function_of_kdd_faster(double d, double kdd, double covariance);
  double dd_2pt_function(double theta_min, double theta_max);
  double dk_2pt_function(double theta_min, double theta_max, int source_bin);
};


class Matter_2D{

 public:

  Matter_2D(Matter* content, string lens_file);
  Matter_2D(Matter* content, string lens_file, string lens_file2); //Marco was here
  Matter_2D(Matter* content, char** lens_file, int number_of_bins, double* z_shift,double* z_spread); //Marco was here
  Matter_2D(Matter* content, char** lens_file, int number_of_bins); //Marco was here

  Matter_2D(Matter* content, string lens_file, string source_file, double dz_lens, double dz_source, vector<double> source_biases);
  Matter_2D(Matter* content, string lens_file, vector<string> source_files, double dz_lens, vector<double> dz_source, vector<double> source_biases);
  ~Matter_2D();
  
  Matter* matter;
  Universe* universe;
    
  void compute_moments(double theta, double* trough_variance, double* trough_skewness);


  void compute_moments(double* theta, int size_theta, vector<vector<double>>* trough_variance, vector<vector<double>>* trough_skewness, long* bins2, int len_bins2,long* bins3, int len_bins3); //Marco was here
  void compute_PDF_and_mean_kappas_from_variable_kappa0(PDF_MODUS PDF_modus, double theta, vector<double> bins, vector<double> *delta_values, vector<double> *PDF, vector<vector<double> > *mean_deltas, vector<vector<vector<double> > > *mean_kappas, double* trough_variance, double* trough_skewness);
  
  void compute_PDF_and_mean_kappas_from_variable_kappa0_Bernardeau(double theta, vector<double> bins, vector<double> *delta_values, vector<double> *PDF, vector<vector<double> > *mean_deltas, vector<vector<vector<double> > > *mean_kappas, double* trough_variance, double* trough_skewness);

  void compute_moments(vector<double> *delta_values, vector<double> *PDF);
  void compute_PofN(double n_bar, double bias, vector<double> *delta_values, vector<double> *PDF, vector<double> *PofN);
  void compute_quantiles(int r_int, string data_file, vector<double> *delta_values, vector<double> *PDF);
  void set_cl_compute(int mute);
  void compute_Cl_kappa_total();
  void compute_Cl_kappa_total_lens(); //Marco here

  void print_2D_stats(string file_name, double bias);
  void print_2D_stats_lens(string file_name, double bias);  //Marco here
  void set_sd(int lensing_A, int lensing_B); //Marco was here
  void set_shift(double* shift);
  void set_IA_params(double A0, double z0,double alpha0);
  void test_wtheta(int n_w, int source_bin, double theta, double delta, vector<double> bins);
  void set_bins(long* bins2, long* bins3,int len_bins2, int len_bins3);
  void set_output(const char **save_out);
  projection_stats stats_2D;

 private:

   int order;
   PDF_MODUS PDF_modus;
   int Cl_compute;
   int sd_lensing_A;
   int sd_lensing_B;
   double eta_min;
   double eta_max;
   double w_min;
   double w_max;
   double* z_shift;
   double* z_spread;
   long* bins2;
   long* bins3;
   int len_bins2;
   int len_bins3;
   double A0;
   double z0;
   double alpha0;
   const char **save_out;

   vector<double> source_biases;


   cosmological_model cosmology;

   vector<ProjectedField> lens_kernels;
   vector<ProjectedField> lens_kernels_2; //Marco was here
   vector<ProjectedField> source_kernels;
   vector<ProjectedField> source_kernels_overlap;
   vector<ProjectedField> source_kernels_density;
 
   void compute_projected_phi_and_G_kappas(int n_w, double theta, vector<double> bins, vector<double> *G_coeffs_projected, vector<vector<vector<double> > > *G_kappa_coeffs_projected, vector<vector<vector<double> > > *G_kappa_coeffs_overlap, vector<vector<vector<double> > > *G_source_density_coeffs, vector<double> *covariances, vector<double> *expectation_of_kdd, vector<double> *covariances_overlap, vector<double> *expectation_of_kdd_overlap, vector<double> *covariances_source_density, vector<double> *expectation_of_kdd_source_density, vector<double> *var_kappa_at_trough_radius);
   void compute_projected_phi_and_G_kappas_Bernardeau(int n_w, double theta, vector<double> bins, vector<double> *G_coeffs_projected, vector<vector<double> > *G_delta_coeffs_projected, vector<vector<vector<double> > > *G_kappa_coeffs_projected, vector<double> *lambda_values, vector<double> *phi_lambda_values, vector<double> *phi_prime_values, vector<vector<double> > *G_delta_values, vector<vector<vector<double> > > *G_kappa_values, double *kd, double *kdd, double *kkd, double *var_kappa_at_trough_radius);
   void get_total_kappa_Cells(vector<vector<double> > * Cl_kappa);
   void get_total_kappa_Cells_lens(vector<vector<double> > * Cl_kappa);

   void set_2D_stats(int n_w, double theta);
   void set_2D_stats(int n_w, double theta, vector<double> bins);
   void set_2D_stats(int n_w, int size_theta, double* theta, long* bins2, int len_bins2,long* bins3, int len_bins3); //Marco was here

   void initialize_2D_stats(double dd, double ddd, vector<double> kd_values, vector<double> kdd_values, vector<double> kk_values, vector<double> kkd_values, vector<double> Cl_delta, vector<vector<double> > Cls_cross, vector<vector<double> > Cls_kappa);
   
};

void Matter_2D::set_sd(int lensing_A, int lensing_B){
    this->sd_lensing_A = lensing_A ;
    this->sd_lensing_B = lensing_B ;
}

void Matter_2D::set_shift(double* shift){
    this->z_shift = shift ;

}

void Matter_2D::set_IA_params(double A0, double z0,double alpha0){
    this->A0=A0;
    this->z0=z0;
    this->alpha0=alpha0;
}

void Matter_2D::set_bins(long* bins2, long* bins3, int len_bins2, int len_bins3){
    this->bins2 = bins2;
    this->bins3 = bins3;
    this->len_bins2 = len_bins2;
    this->len_bins3 = len_bins3;
}
void Matter_2D::set_output(const char **save_out){
    this->save_out = save_out;
}

void Matter_2D::compute_Cl_kappa_total(){
  
  int n = this->stats_2D.Cell_delta.size();
  int n_kappa = this->source_kernels.size();
  
  vector<vector<double> > Cells_kappa_total((n_kappa*(n_kappa+1))/2, vector<double>(n, 0.0));
  
  get_total_kappa_Cells(&Cells_kappa_total);
    
  int kappa_index = 0;
  for(int i = 0; i < n_kappa; i++){
    for(int j = i; j < n_kappa; j++){
      
      //double alpha = this->stats_2D.kk_values_correlated[kappa_index]/(this->stats_2D.kk_values_correlated[kappa_index] + this->stats_2D.kk_values_uncorrelated[kappa_index]);
      double alpha = this->stats_2D.kk_values_correlated[kappa_index]/this->stats_2D.kk_values_total[kappa_index];

      for(int l = 0; l < n; l++){
        if(i == 1) cout << l << '\n';
        this->stats_2D.Cells_kappa[kappa_index][l] *= alpha;
        this->stats_2D.Cells_kappa_uncorrelated[kappa_index][l] = Cells_kappa_total[kappa_index][l] - this->stats_2D.Cells_kappa[kappa_index][l];
      }
      kappa_index++;
    }
  }
}

// Marco here:
void Matter_2D::set_cl_compute(int mute){
this->Cl_compute = mute ;

}
void Matter_2D::compute_Cl_kappa_total_lens(){

  int n = constants::ell_max;
  int n_kappa = this->lens_kernels.size();


  vector<vector<double> > Cells_kappa_total((n_kappa*(n_kappa+1))/2, vector<double>(n, 0.0));



  get_total_kappa_Cells_lens(&Cells_kappa_total);  // in matter 2D_PDF_computation


  /*
  int kappa_index = 0;
  for(int i = 0; i < n_kappa; i++){
    for(int j = i; j < n_kappa; j++){

      //double alpha = this->stats_2D.kk_values_correlated[kappa_index]/(this->stats_2D.kk_values_correlated[kappa_index] + this->stats_2D.kk_values_uncorrelated[kappa_index]);

      double alpha = this->stats_2D.kk_values_correlated[kappa_index]/this->stats_2D.kk_values_total[kappa_index];

      for(int l = 0; l < n; l++){
        if(i == 1) cout << l << '\n';
        this->stats_2D.Cells_kappa[kappa_index][l] *= alpha;
        this->stats_2D.Cells_kappa_uncorrelated[kappa_index][l] = Cells_kappa_total[kappa_index][l] - this->stats_2D.Cells_kappa[kappa_index][l];
      }
      kappa_index++;
    }
  }
  */
}


//Marco here
void  Matter_2D::print_2D_stats_lens(string file_name, double bias=1){
  

    int n = this->stats_2D.Cell_delta.size();
    int n_kappa = this->lens_kernels.size();


    this->compute_Cl_kappa_total_lens(); // routine in


    remove(file_name.c_str());
    FILE* F = fopen(file_name.c_str(), "w");
    fclose(F);
    fstream out;
    out.open(file_name.c_str());
    
    out << scientific << setprecision(10);



    for(int l = 0; l < n; l++){
      out << double(l) << setw(20);

      int kappa_index = 0;
      for(int i = 0; i < n_kappa; i++){
        for(int j = i; j < n_kappa; j++){
          out << this->stats_2D.Cells_kappa_total[kappa_index][l] << setw(20);
          kappa_index++;
        }
      }

      out << '\n';
    }

    out.close();


}


void  Matter_2D::print_2D_stats(string file_name, double bias=1){
  if(this->stats_2D.initialized == 1){

    int n = this->stats_2D.Cell_delta.size();
    int n_kappa = this->source_kernels.size();

    this->compute_Cl_kappa_total_lens();

    remove(file_name.c_str());
    FILE* F = fopen(file_name.c_str(), "w");
    fclose(F);
    fstream out;
    out.open(file_name.c_str());

    out << scientific << setprecision(10);

    out << "# delta_0 = " << this->stats_2D.d0 << '\n';
    for(int i = 0; i < n_kappa; i++)
      out << "# kappa_0" << i << " = " << this->stats_2D.k0_values[i] << '\n';
    out << "#    bias = " << bias << '\n';
    out << "# ell" << setw(20);
    if (bias!=1.0) {
    out << "Cell_delta*b^2 " << setw(20);
    for(int i = 0; i < n_kappa; i++)
      out << "Cell_dk" << i << "*b " << setw(20);
    } else {
    out << "Cell_delta " << setw(20);
    for(int i = 0; i < n_kappa; i++)
      out << "Cell_dk" << i << setw(20);
    }

    for(int i = 0; i < n_kappa; i++){
      for(int j = i; j < n_kappa; j++){
        out << "Cell_k" << i << "k" << j << "_correlated " << setw(20);
      }
    }

    for(int i = 0; i < n_kappa; i++){
      for(int j = i; j < n_kappa; j++){
        out << "Cell_k" << i << "k" << j << "_uncorrelated " << setw(20);
      }
    }

    out << "\n";


    for(int l = 0; l < n; l++){
      out << double(l) << setw(20);
      out << bias*bias*this->stats_2D.Cell_delta[l] << setw(20);
      for(int i = 0; i < n_kappa; i++)
        out << bias*this->stats_2D.Cells_cross[i][l] << setw(20);

      int kappa_index = 0;
      for(int i = 0; i < n_kappa; i++){
        for(int j = i; j < n_kappa; j++){
          out << this->stats_2D.Cells_kappa[kappa_index][l] << setw(20);
          kappa_index++;
        }
      }

      kappa_index = 0;
      for(int i = 0; i < n_kappa; i++){
        for(int j = i; j < n_kappa; j++){
          out << this->stats_2D.Cells_kappa_uncorrelated[kappa_index][l] << setw(20);
          kappa_index++;
        }
      }
      out << '\n';
    }

    out.close();
  }
  else{
    cerr << "WARNING: 2D_stats have not yet been initialized!\nSee function print_2D_stats in Matter_2D.h .\n";
  }
}


double projection_stats::dd_2pt_function(double theta_min, double theta_max){
  
  int ell_max = this->Cell_delta.size();
  double cos_th_min = cos(theta_min);
  double cos_th_max = cos(theta_max);
  double *Pl_th_min = new double[ell_max+2]; 
  double *Pl_th_max = new double[ell_max+2]; 
  gsl_sf_legendre_Pl_array(ell_max+1, cos_th_min, Pl_th_min);
  gsl_sf_legendre_Pl_array(ell_max+1, cos_th_max, Pl_th_max);
  
  double wtheta = 0.0;
  
  if(theta_min > 1.0*constants::arcmin){
  
    for(int i = 1; i < ell_max; i++){
      wtheta += this->Cell_delta[i]*(Pl_th_min[i+1] - Pl_th_min[i-1] - Pl_th_max[i+1] + Pl_th_max[i-1]);
    }
    wtheta /= 4.0*constants::pi*(cos_th_min-cos_th_max);
  }
  else{
    double theta = sqrt(theta_max*theta_min);
    double cos_th = cos(theta);
    double *Pl_th = new double[ell_max+2]; 
    gsl_sf_legendre_Pl_array(ell_max+1, cos_th, Pl_th);
    
    for(int i = 0; i < ell_max; i++){
      wtheta += this->Cell_delta[i]*Pl_th[i]*(2.0*double(i)+1);
    }
    wtheta /= 4.0*constants::pi;
    delete []Pl_th;
  }
  
  
  delete []Pl_th_min;
  delete []Pl_th_max;
  
  return wtheta;
  
}



void return_Legendres_at_one_theta(int n_ell, double cos_theta, vector<double> *Pl){
  
  (*Pl) = vector<double>(n_ell, 0.0);
  
  double *Pl_th_aux = new double[n_ell+3]; 
  
  gsl_sf_legendre_Pl_array(n_ell-1, cos_theta, Pl_th_aux);
  
  for(int l = 0; l < n_ell; l++){
    (*Pl)[l] = Pl_th_aux[l];
  }
  
  delete[] Pl_th_aux;
  
}

double projection_stats::dk_2pt_function(double theta_min, double theta_max, int source_bin){
  
  
  int ell_max = this->Cell_delta.size();
  int n_ell = ell_max+1;
  int n_th = 2;
  
  double cos_th_min = cos(theta_min);
  double cos_th_max = cos(theta_max);
  vector<double> bins(2, 0.0);
  bins[0] = theta_min;
  bins[1] = theta_max;  
  
  double x;
  
  vector<vector<double> > Legendres(n_th, vector<double>(n_ell, 0.0));
  vector<double> Pl_th_aux(n_ell, 0.0); 
  
  for(int th = 0; th < n_th; th++){
    return_Legendres_at_one_theta(n_ell, cos(bins[th]), &Pl_th_aux);
    for(int l = 0; l < n_ell; l++){ 
      Legendres[th][l] = Pl_th_aux[l];
    }
  }
  
  double normalisation = cos_th_min - cos_th_max;
  double Legendre_new;
  vector<double> Pl_th(n_ell, 0.0);
  double x1 = cos_th_min;
  double x2 = cos_th_max;
  for(int l = 1; l < n_ell-1; l++){
    Legendre_new = -2.0/double(2*l+1)*(Legendres[0][l+1] - Legendres[0+1][l+1]);
    Legendre_new += double(2-l)*(x1*Legendres[0][l] - x2*Legendres[0+1][l]);
    Legendre_new += double((2*l+1)*l+2)/double(2*l+1)*(Legendres[0][l-1] - Legendres[0+1][l-1]);
    Legendre_new /= normalisation*double(l*(l+1));
    Pl_th[l] = Legendre_new;
  }
  
  
  double xi = 0.0;
  double ell;
  double Legendre_factor;
  
  for(int l = 0; l < n_ell; l++){
    ell = double(l);
    Legendre_factor = Pl_th[l];
    xi += (2.0*ell + 1.0)/(4.0*pi)*Legendre_factor*this->Cells_cross[source_bin][l];
  }
    
  return xi;
    
}



