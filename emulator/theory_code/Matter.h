#include <thread>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

class Matter {

 public:

  Matter(Universe* uni, double s3_enhance);
  ~Matter();
  
  Universe* universe;

  void print_Newtonian_growth_factor(string file_name);
  
  void compute_lognormal_PDF_only(double R1, double eta, vector<double> *G_coeffs);
  void compute_lognormal_old(double R1, double eta, vector<double> bins, vector<double> *G_coeffs, vector<vector<double> > *G_kappa_coeffs);
  void compute_variance_and_skewness(double R1, double eta, vector<double> *G_coeffs);


  void compute_variance_and_skewness_arr(int size_theta, vector<double> R_arr, double eta, vector<vector<double> > *G_coeffs_at_w_arr); //Marco was here
  void compute_lognormal_final(double R1, double eta, vector<double> *bins, vector<double> *G_coeffs, vector<vector<double> > *G_kappa_coeffs);
  void compute_Bernardeau_final(double R1, double eta, vector<double> bins, vector<double> *G_coeffs, vector<vector<double> > *G_kappa_coeffs, vector<double> *lambda_of_w, vector<double> *phi_lambda_of_w, vector<double> *phi_prime_lambda_of_w, vector<vector<double> > *G_kappa_of_w);
  
  void return_delta_NL_of_delta_L(double eta, vector<double> *delta_L_values, vector<double> *delta_NL_values);
  void return_delta_NL_of_delta_L_and_dF_ddelta(double eta, vector<double> *delta_L_values, vector<double> *delta_NL_values, vector<double> *delta_NL_prime_values);
  
  void update_linear_power_spectrum(BINNING binning, vector<double> k_values, vector<double> P_k_lin_values);
  
  double Newtonian_linear_power_spectrum(double k, double e);
  double Newtonian_linear_power_spectrum(double k);
  double transfer_function_at(double k);
  double current_P_NL_at(double k);
  double current_P_L_at(double k);
  
  vector<double> P_L(double e);
  vector<double> P_NL(double e);
  vector<double> return_wave_numbers();
  vector<double> log_top_hat_radii;
  vector<double> top_hat_cylinder_variances;

  void set_spherical_collapse_evolution_of_delta(double z_min, double z_max, int n_time);
  void set_permanent_Legendres(int l_max, double theta_T);
  void set_permanent_Legendres_null(int l_max, double theta_T); //marco here
  void set_permanent_Legendres_array(int l_max, double* theta_T, int size_theta); //Marco here
  void set_permanent_Legendres(int l_max, double theta_T, vector<double> bins);
  void set_linear_Legendres(int modus, double theta);
  void update_Cell(double dw, double weight1, double weight2, vector<double>* Cell);
  void update_Cell(double w, double dw, double weight1, double weight2, vector<double>* Cell);
  void set_d23_output(char ** outp_d23);
  void return_out_d23(int which_one, char** mutec);
  void load_masks(double* xxx,int len_mask,int size_theta,double fact_area);
  void read_pix_func(string name_CL_pix_mask, int len_pix_mask); //Marco here
  void set_OWL(int OWL, double r_OWL,string DM_FILENAMEL,string U_FILENAME,string L_FILENAME,string powz,string powk, string powele);
  void set_NL_p(int NL_p);
  void set_masks(double* mask_E, double* mask_B,int len_mask,double fact_area, double* Cl_pix_mask, int len_pix_mask); //Marco here
  void print_power_spectra(double z);
  int order;
  
  double variance_of_matter_within_R(double R);
  double variance_of_matter_within_R_2D(double R);
  void variance_of_matter_within_R_2D_arr(int size_theta, vector<double> *variance_array); //Marco was here
  void variance_of_matter_within_R_2D_NL_arr(int size_theta, vector<double> *variance_array); //Marco was here
  void variance_of_matter_within_R_2D_NL_arr_mod(int size_theta, vector<double> *variance_array_a,vector<double> *variance_array_b,vector<double> *variance_array_c, double w);
  void variance_of_matter_within_R_2D_arr_mod(int size_theta, vector<double> *variance_array_a,vector<double> *variance_array_b,vector<double> *variance_array_c, double w);
  double variance_of_matter_within_R_2D();
  double variance_of_matter_within_R_2D_NL();
  double covariance_of_matter_within_R_2D();
  double dcov1_at() const;
  double dcov2_at() const;
  double covariance_of_matter_within_R_2D_NL(int bin);
  double covariance_of_matter_within_R_2D_NL();
  
  double return_Delta_prime_of_eta(double eta);
  double  growth(double eta);

 private:
   
  int number_of_entries_Newton;
  int ell_max;

  double eta_initial;
  double eta_final;
  double norm;
  double top_hat_radius;
  double second_top_hat_radius;
  double S3_enhance;
      
  vector<double> eta_Newton;   
  vector<double> eta_NL;   
  vector<double> Newtonian_growth_factor_of_delta;
  vector<double> Newtonian_growth_factor_of_delta_prime;
  vector<double> Newtonian_growth_factor_of_delta_prime_prime;
  vector<double> Newtonian_growth_factor_of_delta_prime_prime_prime;
  vector<double> Newtonian_growth_factor_second_order;
  vector<double> Newtonian_growth_factor_second_order_prime;

  vector<double> wave_numbers;
  vector<double> log_wave_numbers;
  vector<double> transfer_function;

  vector<double> delta_values;
  vector<vector<double> > spherical_collapse_evolution_of_delta;
  vector<vector<double> > spherical_collapse_evolution_of_delta_ddelta;
  vector<double> F_prime_of_eta;
  vector<double> F_prime_prime_of_eta;



  // MARCO WAS HERE: these are defined for the array case.
  vector<vector<double>> Atheta_trough_Legendres;
  vector<vector<double>> Adtheta_trough_Legendres_dtheta_trough;



  vector<double> current_P_delta_NL_in_trough_format;
  vector<double> current_P_delta_NL_in_trough_format_times_trough_legendres;
  vector<double> current_P_delta_L_in_trough_format;
  vector<double> current_P_delta_L_in_trough_format_times_trough_legendres;
  vector<double> current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough;
  vector<double> theta_trough_Legendres;
  vector<double> dtheta_trough_Legendres_dtheta_trough;
  vector<vector<double> > bin_Legendres;
  vector<vector<double> > dbin_Legendres_dtheta_bin;
  vector<double> current_theta1_linear_Legendres;
  vector<double> current_theta2_linear_Legendres;
  vector<double> current_theta1_linear_Legendres_prime;
  vector<double> current_theta2_linear_Legendres_prime;
   
  vector<double> values_for_covariance_computation;
   
  void load_Mead_power();
  vector<double> Mead_z_values;
  vector<double> Mead_k_values;
  vector<vector<double> > Mead_power_spectrum;
   
  void load_CAMB_power();
  vector<double> CAMB_k_values;
  vector<double> CAMB_ln_k_values;
  vector<double> CAMB_power_spectrum;

  vector<vector<double>> maskE; //Marco was here
  vector<vector<double>> maskB; //Marco was here
  double r_OWL;
  int OWL;
  int NL_p;
  vector<vector<double> > Pkz;
  vector<vector<double> > Pkzb;
  vector<vector<double> > Pkzt;
  vector<vector<double> > ell;
  vector<double> pz;
  vector<double> pk;

  double l_nl;
  double Dp;
  double* maskdue;
  //arma::cube maskE_armad;
  //arma::rowvec maskE_vec;
  //double* maskE;
  //double* maskB;
  double fact_area; //Marco was here
  int len_mask; //Marco was here
  vector<double> Cl_mask; //Marco was here
  //double* Cl_mask; //Marco was here
  int len_pix_mask; //Marco was here
  char ** d23_output;

  cosmological_model cosmology;
      
  void set_matter_content();
  void set_wave_numbers();
  void set_transfer_function_Eisenstein_and_Hu();
  void set_transfer_function_Bond_and_Efstathiou();
  void set_cylinder_variances();
   
  void prepare_power_spectra_in_trough_format(double w);
  void prepare_power_spectra_in_trough_format_2(double w); //MARCO

  void initialize_linear_growth_factor_of_delta(double D, double D_prime);
  void initialize_up_to_second_order_growth_factor_of_delta(double D, double D_prime);
  void compute_G_and_tau(double R, vector<double> *tau_output, vector<double> *G_output, vector<double> *G_prime_output, double *y_critical);
   
  /*******************************************************************************************************************************************************
   * These functions and variables are for the Smith_et_al fitting formula
   *******************************************************************************************************************************************************/

  double current_k_non_linear;
  double current_n_eff;
  double current_C_sm;
  double current_scale;
   
  vector<double> current_P_NL;
  vector<double> current_P_L;
   
  double sig_sq(double R, double e);
  vector<double> c_and_n_NL(double R, double e);

  double k_NL(double k_min, double k_max, double e);

  double Delta_Q_sq(double k, double e);
  double Delta_H_sq(double k);
  double Delta_H_prime_sq(double k);
  double P_NL_at(double k, double e);
   
  void sig_sq_derivs(vector<double> (*a), vector<double> (*dadt));
  void norm_derivs(vector<double> (*a), vector<double> (*dadt));
  void c_and_n_derivs(vector<double> (*a), vector<double> (*dadt));

  double f_1, f_2, f_3, mun, nun, betan, alphan;
  double current_r_sq;
   
};

double Matter::growth(double eta){
    return interpolate_neville_aitken(eta, &(this->eta_Newton), &(this->Newtonian_growth_factor_of_delta), this->order);
}
void Matter::load_Mead_power(){
  
  int nz = 100;
  int nk = 200;
  
  double dummy;
  double current_k;
  
  Mead_z_values = vector<double>(nz, 0.0);
  Mead_k_values = vector<double>(nk, 0.0);
  Mead_power_spectrum = vector<vector<double> >(nz, vector<double>(nk, 0.0));
  
  fstream in("Data/HMCODE_results/power.dat");
  in >> dummy;
  for(int i = 0; i < nz; i++){
    in >> Mead_z_values[i];
  }
  for(int j = 0; j < nk; j++){
    for(int i = -1; i < nz; i++){
      if(i == -1){
        in >> current_k;
        Mead_k_values[j] = current_k*c_over_e5;
      }
      else{
        in >> Mead_power_spectrum[i][j];
        Mead_power_spectrum[i][j] /= pow(current_k, 3.0)/(2.0*constants::pi*constants::pi);
      }
    }
  }
  
  in.close();
  
}


void Matter::load_CAMB_power(){
  
  int nk = 205;
  
  double dummy;
  double current_k;
  
  string headline;
  
  CAMB_k_values = vector<double>(nk, 0.0);
  CAMB_ln_k_values = vector<double>(nk, 0.0);
  CAMB_power_spectrum = vector<double>(nk, 0.0);
  
  fstream ink("Data/CAMB_results/k_h.txt");
  fstream inp("Data/CAMB_results/p_k.txt");
  
  getline(ink, headline);
  getline(inp, headline);
  
  for(int i = 0; i < nk; i++){
    ink >> current_k;
    CAMB_k_values[i] = current_k*c_over_e5;
    CAMB_ln_k_values[i] = log(current_k*c_over_e5);
  }
  
  for(int i = 0; i < nk; i++){
    inp >> CAMB_power_spectrum[i];
  }
  
  ink.close();
  inp.close();
  
}



void Matter::update_Cell(double w, double dw, double weight_1, double weight_2, vector<double>* Cell){
  this->prepare_power_spectra_in_trough_format_2(w);
  for(int ell = 1; ell <2048; ell++){//this->ell_max; ell++){
    (*Cell)[ell] += this-> current_P_delta_NL_in_trough_format[ell]*weight_1*weight_2*dw;

  }

}

void Matter::update_Cell(double dw, double weight_1, double weight_2, vector<double>* Cell){
  for(int ell = 1; ell < 2048; ell++){//this->ell_max; ell++){
    (*Cell)[ell] += this->current_P_delta_NL_in_trough_format[ell]*weight_1*weight_2*dw;
  }
  
}

void Matter::set_d23_output(char ** outp_d23){
    this->d23_output = outp_d23;
}

void Matter::return_out_d23(int which_one, char** mutec){
    *mutec = this->d23_output[which_one];
}

//Marco was here
void Matter::load_masks(double* xxx,int len_mask,int size_theta,double fact_area){
    clock_t start = clock();
    this->len_mask = len_mask;
    this->fact_area = fact_area;
   this->maskdue = xxx;
   cout << "load mask into memory : " << clock() - start << "ms \n";

}

//Marco was here
void Matter::read_pix_func(string name_CL_pix_mask, int len_pix_mask){

  if (len_pix_mask != 0){

  fstream inE;
  inE.open(name_CL_pix_mask);

  double dummy;
  vector<double> maskE_dummy(0, 0.0);

  while(inE.good()){
    inE >> dummy;
    maskE_dummy.push_back(dummy);
    }


  inE.close();

  this->Cl_mask = maskE_dummy;

  this->len_pix_mask = len_pix_mask;

  }else{
  vector<double> maskE_dummy(0, 0.0);

  for(int ell = 1; ell < 10001; ell++){
  maskE_dummy.push_back(1.);
  }
  this->Cl_mask = maskE_dummy;
  this->len_pix_mask = ell_max;
  }
}

void Matter:: set_NL_p(int NL_p){
    this->NL_p = NL_p;
}

void Matter::set_OWL( int OWL,double r_OWL,string DM_FILENAMEL,string U_FILENAME,string L_FILENAME,string powz, string powk, string powele){
  this->r_OWL =r_OWL;
  this->OWL=OWL;
  fstream inz;
  inz.open(powz);

  double dummy;
  vector<double> z(0, 0.0);
  vector<double> k(0, 0.0);

  while(inz.good()){
    inz >> dummy;

    z.push_back(dummy);
    }

  inz.close();

  fstream ink;
  ink.open(powk);

  double dummyk;

  while(ink.good()){
    ink >> dummyk;

    k.push_back(dummyk);
    }

  ink.close();





  fstream inE;
  inE.open(DM_FILENAMEL);


  vector<double> Pkz(0, 0.0);

  while(inE.good()){
    inE >> dummy;

    Pkz.push_back(dummy);
    }

  inE.close();


  fstream inB;
  inB.open(U_FILENAME);

  double dummyb;
  vector<double> Pkzb(0, 0.0);

  while(inB.good()){
    inB >> dummyb;
    Pkzb.push_back(dummyb);
    }

  inB.close();


  fstream inT;

  inT.open(L_FILENAME);

  double dummyt;
  vector<double> Pkzt(0, 0.0);

  while(inT.good()){
    inT >> dummyt;
    Pkzt.push_back(dummyt);
    }

  inT.close();

  this->Pkz  = vector<vector<double> >(k.size(), vector<double>(z.size(), 0.0));
  this->Pkzb  = vector<vector<double> >(k.size(), vector<double>(z.size(), 0.0));
  this->Pkzt  = vector<vector<double> >(k.size(), vector<double>(z.size(), 0.0));

  for(int i =0; i<k.size();i++){
    for(int j = 0; j<z.size();j++){


        this->Pkzt[i][j] = Pkzt[j+i*z.size()];
        this->Pkzb[i][j] = Pkzb[j+i*z.size()];
        this->Pkz[i][j] = Pkz[j+i*z.size()];

    }
  }

  this->pz = z;
  this->pk = k;
}
