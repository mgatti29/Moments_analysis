
/*
 * Preparing Legendre polynomial for variance and covariance
 * computations. There is one set of polynomial values for the
 * troughs and one for the bin. Each set receives an additional
 * factor sqrt(pi)/A.
 */


void Matter::set_permanent_Legendres(int l_max, double theta_T, vector<double> bins){
  //cout << "perm. Legendres set.\n";
  this->ell_max = l_max;

  this->current_P_delta_NL_in_trough_format = vector<double>(l_max, 0.0);
  this->current_P_delta_NL_in_trough_format_times_trough_legendres = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format_times_trough_legendres = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough = vector<double>(l_max, 0.0);
  this->theta_trough_Legendres = vector<double>(this->ell_max, 0.0);
  this->dtheta_trough_Legendres_dtheta_trough = vector<double>(this->ell_max, 0.0);
  this->current_theta1_linear_Legendres = vector<double>(this->ell_max, 0.0);
  this->current_theta2_linear_Legendres = vector<double>(this->ell_max, 0.0);
  this->current_theta1_linear_Legendres_prime = vector<double>(this->ell_max, 0.0);
  this->current_theta2_linear_Legendres_prime = vector<double>(this->ell_max, 0.0);
  this->bin_Legendres = vector<vector<double> >(bins.size(), vector<double>(this->ell_max+1, 0.0));
  this->dbin_Legendres_dtheta_bin = vector<vector<double> >(bins.size(), vector<double>(this->ell_max+1, 0.0));
  

  
  double cos_th = cos(theta_T);
  double sin_th = sin(theta_T);
  double *Pl_th = new double[l_max+2]; 
        gsl_sf_legendre_Pl_array(l_max+1, cos_th, Pl_th);
  double A;
  give_area_from_cosines(1.0, cos_th, &A);
  for(int ell = 1; ell < l_max; ell++){
    this->theta_trough_Legendres[ell] = (Pl_th[ell+1]-Pl_th[ell-1])*sqrt(constants::pi/(2.0*double(ell)+1.0))/A;
    this->dtheta_trough_Legendres_dtheta_trough[ell] = -sin_th*Pl_th[ell]*sqrt(constants::pi*(2.0*double(ell)+1.0))/A;
    this->dtheta_trough_Legendres_dtheta_trough[ell] += this->theta_trough_Legendres[ell]*(-constants::pi2*sin_th/A);
  }
  
  for(int b = 0; b < bins.size(); b++){
    cos_th = cos(bins[b]);
    sin_th = sin(bins[b]);
    gsl_sf_legendre_Pl_array(l_max+1, cos_th, Pl_th);
    give_area_from_cosines(1.0, cos_th, &A);
    for(int ell = 1; ell < l_max; ell++){
      this->bin_Legendres[b][ell] = (Pl_th[ell+1]-Pl_th[ell-1])*sqrt(constants::pi/(2.0*double(ell)+1.0))/A;
      this->dbin_Legendres_dtheta_bin[b][ell] = -sin_th*Pl_th[ell]*sqrt(constants::pi*(2.0*double(ell)+1.0))/A;
      this->dbin_Legendres_dtheta_bin[b][ell] += this->bin_Legendres[b][ell]*(-constants::pi2*sin_th/A);
    }
  }

      delete []Pl_th;
  
}

//Marco was here
void Matter::set_permanent_Legendres_array(int l_max, double* theta_T, int size_theta){

  this->ell_max = l_max;

  /*
  this->Acurrent_P_delta_NL_in_trough_format = vector<vector<double>>(size_theta, vector<double> (l_max, 0.0));
  this->Acurrent_P_delta_NL_in_trough_format_times_trough_legendres = vector<vector<double>>(size_theta, vector<double> (l_max, 0.0));
  this->Acurrent_P_delta_L_in_trough_format = vector<vector<double>>(size_theta, vector<double> (l_max, 0.0));
  this->Acurrent_P_delta_L_in_trough_format_times_trough_legendres = vector<vector<double>>(size_theta, vector<double> (l_max, 0.0));
  this->Acurrent_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough = vector<vector<double>>(size_theta, vector<double> (l_max, 0.0));
  */
  this->current_P_delta_NL_in_trough_format = vector<double>(l_max, 0.0);
  this->current_P_delta_NL_in_trough_format_times_trough_legendres = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format_times_trough_legendres = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough = vector<double>(l_max, 0.0);
  this->Atheta_trough_Legendres = vector<vector<double>>(size_theta, vector<double> (this->ell_max, 0.0));
  this->Adtheta_trough_Legendres_dtheta_trough = vector<vector<double>>(size_theta, vector<double> (this->ell_max, 0.0));

  for(int i = 0; i < size_theta; i++){
    double cos_th = cos(theta_T[i]);
    double sin_th = sin(theta_T[i]);
    double *Pl_th = new double[l_max+2];
    gsl_sf_legendre_Pl_array(l_max+1, cos_th, Pl_th);
    double A;
    give_area_from_cosines(1.0, cos_th, &A);
    if (theta_T[i] == 0.0){
        for(int ell = 1; ell < l_max; ell++){
            this->Atheta_trough_Legendres[i][ell] = sqrt(constants::pi*(2.0*double(ell)+1.0))/constants::pi2;
            this->Adtheta_trough_Legendres_dtheta_trough[i][ell] = 0. ;
            this->Adtheta_trough_Legendres_dtheta_trough[i][ell] += 0. ;
        }

    }else{
        for(int ell = 1; ell < l_max; ell++){

            this->Atheta_trough_Legendres[i][ell] = (Pl_th[ell+1]-Pl_th[ell-1])*sqrt(constants::pi/(2.0*double(ell)+1.0))/A;
            this->Adtheta_trough_Legendres_dtheta_trough[i][ell] = -sin_th*Pl_th[ell]*sqrt(constants::pi*(2.0*double(ell)+1.0))/A;
            this->Adtheta_trough_Legendres_dtheta_trough[i][ell] += this->Atheta_trough_Legendres[i][ell]*(-constants::pi2*sin_th/A);


        }
    }

    delete []Pl_th;

    }
  }



void Matter::set_permanent_Legendres(int l_max, double theta_T){
  
  this->ell_max = l_max;

  this->current_P_delta_NL_in_trough_format = vector<double>(l_max, 0.0);
  this->current_P_delta_NL_in_trough_format_times_trough_legendres = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format_times_trough_legendres = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough = vector<double>(l_max, 0.0);
  this->theta_trough_Legendres = vector<double>(this->ell_max, 0.0);
  this->dtheta_trough_Legendres_dtheta_trough = vector<double>(this->ell_max, 0.0);
  this->current_theta1_linear_Legendres = vector<double>(this->ell_max, 0.0);
  this->current_theta2_linear_Legendres = vector<double>(this->ell_max, 0.0);
  this->current_theta1_linear_Legendres_prime = vector<double>(this->ell_max, 0.0);
  this->current_theta2_linear_Legendres_prime = vector<double>(this->ell_max, 0.0);  

  
  double cos_th = cos(theta_T);
  double sin_th = sin(theta_T);
  double *Pl_th = new double[l_max+2]; 
  gsl_sf_legendre_Pl_array(l_max+1, cos_th, Pl_th);
  double A;
  give_area_from_cosines(1.0, cos_th, &A);
  for(int ell = 1; ell < l_max; ell++){

    this->theta_trough_Legendres[ell] = (Pl_th[ell+1]-Pl_th[ell-1])*sqrt(constants::pi/(2.0*double(ell)+1.0))/A;
    this->dtheta_trough_Legendres_dtheta_trough[ell] = -sin_th*Pl_th[ell]*sqrt(constants::pi*(2.0*double(ell)+1.0))/A;
    this->dtheta_trough_Legendres_dtheta_trough[ell] += this->theta_trough_Legendres[ell]*(-constants::pi2*sin_th/A);
  }


  delete []Pl_th;  
}

//Marco was here
void Matter::set_permanent_Legendres_null(int l_max, double theta_T){

  this->ell_max = l_max;

  this->current_P_delta_NL_in_trough_format = vector<double>(l_max, 0.0);
  this->current_P_delta_NL_in_trough_format_times_trough_legendres = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format_times_trough_legendres = vector<double>(l_max, 0.0);
  this->current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough = vector<double>(l_max, 0.0);
  this->theta_trough_Legendres = vector<double>(this->ell_max, 0.0);
  this->dtheta_trough_Legendres_dtheta_trough = vector<double>(this->ell_max, 0.0);
  this->current_theta1_linear_Legendres = vector<double>(this->ell_max, 0.0);
  this->current_theta2_linear_Legendres = vector<double>(this->ell_max, 0.0);
  this->current_theta1_linear_Legendres_prime = vector<double>(this->ell_max, 0.0);
  this->current_theta2_linear_Legendres_prime = vector<double>(this->ell_max, 0.0);


  double cos_th = cos(theta_T);
  double sin_th = sin(theta_T);
  double *Pl_th = new double[l_max+2];
  gsl_sf_legendre_Pl_array(l_max+1, cos_th, Pl_th);
  double A;
  give_area_from_cosines(1.0, cos_th, &A);
  for(int ell = 1; ell < l_max; ell++){

    this->theta_trough_Legendres[ell] = sqrt(constants::pi*(2.0*double(ell)+1.0))/constants::pi2;
    this->dtheta_trough_Legendres_dtheta_trough[ell] = 0. ;
    this->dtheta_trough_Legendres_dtheta_trough[ell] += 0. ;
  }

  delete []Pl_th;
}


//MARCO WAS HERE





void Matter::prepare_power_spectra_in_trough_format_2(double w){

  double eta = this->universe->eta_at_a(1.0) - w;
  double Tell;

  double ell_double;
  double ell_double_plus_half;
  double ln_k;
  double z00;
  double Pdm;
  double Pdmu;
  double Pdml;

  double prefactor = inverse_c_over_e5*inverse_c_over_e5*inverse_c_over_e5/pow(w, 2);
  double one_over_w = 1.0/w;
  this->current_P_L = this->P_L(eta);
  this->current_P_NL = this->P_NL(eta);






  double fact_agn;
  double chi;
  double k_log10;
  for(int ell = 1; ell < this->ell_max; ell++){

    ell_double = double(ell);
    ell_double_plus_half = ell_double + 0.5;
    ln_k = log(ell_double_plus_half*one_over_w);
    Tell = ((ell_double+2.0)*(ell_double+1.0)*ell_double*(ell_double-1.0))/pow(ell_double_plus_half, 4.0)*prefactor;



    //ADD Taka window function ***********************************************

    double c1 = 0.00095171;
    double c2 = 0.0051543;
    double a1 = 1.3063;
    double a2 = 1.1475;
    double a3 = 0.62793;
    z00 =1.0/this->universe->a_at_eta(this->universe->eta_at_a(1.0) - w) - 1.0;

    double kk = ell_double_plus_half/(w*4344.*0.7);

    double muth=1;
    if (this->OWL ==2 or this->OWL ==3){
       
        muth = pow((1.+c1*pow( kk,-a1)),a1)/ pow((1.+c2*pow( kk,-a2)),a3);
        //cout<<muth<<"   "<<ell_double_plus_half<< "   "<<kk<< "   "<<z00<<endl;
        Tell ;
        
    }

    //ADD OWL ****************************************************************


    k_log10= log10(ell_double_plus_half/(w*4344.*0.7));
    if (this->OWL ==1 or this->OWL ==2){
    if( (k_log10>(this->pk)[0]) and (k_log10<(this->pk)[(this->pk).size()-1])){

        Pdm = interpolate_neville_aitken_grid(k_log10, z00, &(this->pk), &(this->pz),&(this->Pkz), 3,3);
        Pdmu = interpolate_neville_aitken_grid(k_log10, z00, &(this->pk), &(this->pz),&(this->Pkzb), 3,3);
        Pdml = interpolate_neville_aitken_grid(k_log10, z00, &(this->pk), &(this->pz),&(this->Pkzt), 3,3);

        fact_agn = this->r_OWL * pow(10,Pdmu)/ pow(10,Pdm) + (1 - this->r_OWL) *  pow(10,Pdml)/ pow(10,Pdm);
        //cout<< ell_double << "   "<< fact_agn<<"  " << pow(10,Pdmu)/ pow(10,Pdm)<<"  "<< pow(10,Pdml)/ pow(10,Pdm)<<"  "<<pow(10,Pdm)<<endl;
    }else{
        fact_agn = 1.;
    }


    this->current_P_delta_L_in_trough_format[ell] = Tell*this->current_P_L_at(ln_k)*fact_agn*muth;///pow(exp(ln_k),2.0);
    this->current_P_delta_NL_in_trough_format[ell] = Tell*this->current_P_NL_at(ln_k)*fact_agn*muth;
    }else{
    this->current_P_delta_L_in_trough_format[ell] = Tell*this->current_P_L_at(ln_k)*muth;///pow(exp(ln_k),2.0);
    this->current_P_delta_NL_in_trough_format[ell] = Tell*this->current_P_NL_at(ln_k)*muth;

    }
  }
  // define non linear scale:
  int Aswitch =0 ;
  double dcheck=0;
  this->l_nl = this->ell_max;
  for(int ell = 1; ell < this->ell_max; ell++){
    double Pnl = this->current_P_delta_L_in_trough_format[ell];
    double kk = ell/(w*4344.*0.7);
    ell_double = double(ell);
    ell_double_plus_half = ell_double + 0.5;
    ln_k = log(ell_double_plus_half*one_over_w);
    //Pnl = current_P_L_at(ln_k)*prefactor;;
    double limm = Pnl*pow(ell/w,3)/(4*pi);
    //cout<< <<" "<<z00  <<"  "<<ell<<"  "<<1./kk<<endl;
    //cout<<Pnl*pi*pow(4344.*0.7,3)<<" "<<  Pnl*pow(ell,3)/(2*pi)/w<<" "<<z00  <<"  "<<1./kk<<"  "<<ell<<" "<<w<<endl;

    /*
    the right comparison should be:
        = 1 Pnl*pow(ell,3)/(2*pi)/w
    */
    //if ((Aswitch==0) and (1./kk<3.)){
    if ((Aswitch==0) and ( Pnl*pow(ell,3)/(2*pi*pi)/(w)>1.)){
        this->l_nl = ell;
        this->l_nl=this->current_k_non_linear*w;
        dcheck = 1./kk;
        Aswitch=1;
    }



  }

  double Dp = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
  this->Dp = Dp;
//  cout <<this->l_nl<<" "<<z00<<"    "<<w<<endl;
}




void Matter::print_power_spectra(double z){

  double w = this->universe->eta_at_a(1.0) - this->universe->eta_at_a(1.0/(1.0+z));
  double eta = this->universe->eta_at_a(1.0) - w;
  double Tell;

  double ell_double;
  double ell_double_plus_half;
  double ln_k;


  double prefactor = inverse_c_over_e5*inverse_c_over_e5*inverse_c_over_e5/pow(w, 2);
  double one_over_w = 1.0/w;
  this->current_P_L = this->P_L(eta);
  this->current_P_NL = this->P_NL(eta);

   FILE* F = fopen("./pk.txt", "w");
   fclose(F);
   fstream out;
   out.open("./pk.txt");







  for(int ell = 1; ell < 10000; ell++){
    ell_double = double(ell);
    ell_double_plus_half = ell_double + 0.5;

    double mute_k = log(ell_double_plus_half);
    double ln_k = log(ell_double_plus_half*one_over_w/100.);
    double mute_lin = this->current_P_L_at(ln_k);
    double mute_nl = this->current_P_NL_at(ln_k);
    Tell = ((ell_double+2.0)*(ell_double+1.0)*ell_double*(ell_double-1.0))/pow(ell_double_plus_half, 4.0)*prefactor;
    out<<  ell_double_plus_half*one_over_w <<" "  << mute_lin*Tell << "   " <<mute_nl*Tell <<endl;

   }
  out.close();
}


void Matter::prepare_power_spectra_in_trough_format(double w){
  
  double eta = this->universe->eta_at_a(1.0) - w;
  double Tell;
  /*
    TOM WAS HERE
    -Om_m, Om_l, scale, H, H_prime, H_prime_prime and ws are not used at all. They are being commented out for optimization.
  */
  /*
    double Om_m = this->cosmology.Omega_m;
    double Om_l = this->cosmology.Omega_L;
    double scale = this->universe->a_at_eta(eta);
    double H = this->universe->H_at_eta(eta);
    double H_prime = this->universe->H_prime_at_eta(eta);
    
    //TOM WAS HERE
    //- Moved H out of the ()

    double H_prime_prime = 0.5*H*(Om_m/scale + 4.0*Om_l*scale*scale);
    //double H_prime_prime = 0.5*(Om_m*H/scale + 4.0*Om_l*scale*scale*H);
    double ws = this->universe->eta_at_a(1.0) - this->universe->eta_at_a(1.0/(2.025));
    */


  /*double f = pow(w, 1.5);
    double f_prime = 1.5*pow(w, 0.5);
    double f_prime_prime = 1.5*0.5*pow(w, -0.5);
    double f_prime_prime_prime = -1.5*0.5*0.5*pow(w, -1.5);
    double f1 = pow(w, -1.5) - pow(w, -0.5)/ws;
    double f1_prime = -1.5*pow(w, -2.5) + 0.5*pow(w, -1.5)/ws;
    double f1_prime_prime = 1.5*2.5*pow(w, -3.5) - 0.5*1.5*pow(w, -2.5)/ws;
    double f1_prime_prime_prime = -1.5*2.5*3.5*pow(w, -4.5) + 0.5*1.5*2.5*pow(w, -3.5)/ws;
  
    double f2 = 1.0/scale;
    double f2_prime = -(-H/scale);
    double f2_prime_prime = (-H_prime/scale + H*H/scale);
    double f2_prime_prime_prime = -(-H_prime_prime/scale + 3.0*H*H_prime/scale - H*H*H/scale);
  
    double f = f1*f2;
    double f_prime = f1_prime*f2 + f2_prime*f1;
    double f_prime_prime = f1_prime_prime*f2 + 2.0*f1_prime*f2_prime + f2_prime_prime*f1;
    double f_prime_prime_prime = f1_prime_prime_prime*f2 + 3.0*f1_prime_prime*f2_prime + 3.0*f1_prime*f2_prime_prime + f1*f2_prime_prime_prime;
  

    double D = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta, this->order);
    double dDdw = -interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta_prime, this->order);
    double d2Ddw2 = interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta_prime_prime, this->order);
    double d3Ddw3 = -interpolate_neville_aitken(eta, &this->eta_Newton, &this->Newtonian_growth_factor_of_delta_prime_prime_prime, this->order);
  
    double W = f*D;
    double W_prime_prime = f_prime_prime*D + 2.0*f_prime*dDdw + f*d2Ddw2;
    double W_prime_prime_prime = f_prime_prime_prime*D + 3.0*f_prime_prime*dDdw + 3.0*f_prime*d2Ddw2 + f*d3Ddw3;
  
    double rescaling_part = (pow(w, 2)*W_prime_prime + pow(w, 3)/3.0*W_prime_prime_prime)/W;
    double rescaling;*/
  
  double ell_double;
  double ell_double_plus_half;
  double ln_k;
  /*
    TOM WAS HERE
    - To get rid of divisions /c_over_e5 is replaced with *inverse_c_over_e5
  */
  //double prefactor = 1.0/pow(c_over_e5, 3)/pow(w, 2);
  double prefactor = inverse_c_over_e5*inverse_c_over_e5*inverse_c_over_e5/pow(w, 2);
  double one_over_w = 1.0/w;
  this->current_P_L = this->P_L(eta);
  this->current_P_NL = this->P_NL(eta);  
  
  /*this->current_P_delta_NL_in_trough_format = vector<double>(this->ell_max, 0.0);
    this->current_P_delta_NL_in_trough_format_times_trough_legendres = vector<double>(this->ell_max, 0.0);
    this->current_P_delta_L_in_trough_format = vector<double>(this->ell_max, 0.0);
    this->current_P_delta_L_in_trough_format_times_trough_legendres = vector<double>(this->ell_max, 0.0);
    this->current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough = vector<double>(this->ell_max, 0.0);
  */
  for(int ell = 1; ell < this->ell_max; ell++){ 
    ell_double = double(ell);
    ell_double_plus_half = ell_double + 0.5;
    ln_k = log(ell_double_plus_half*one_over_w);
    Tell = ((ell_double+2.0)*(ell_double+1.0)*ell_double*(ell_double-1.0))/pow(ell_double_plus_half, 4.0)*prefactor;
    //rescaling = 1.0 - rescaling_part/pow(double(ell)+0.5, 2);
    this->current_P_delta_L_in_trough_format[ell] = Tell*this->current_P_L_at(ln_k);///pow(exp(ln_k),2.0);
    this->current_P_delta_L_in_trough_format_times_trough_legendres[ell] = this->current_P_delta_L_in_trough_format[ell]*this->theta_trough_Legendres[ell];
    this->current_P_delta_L_in_trough_format_times_dtrough_legendres_dtheta_trough[ell] = this->current_P_delta_L_in_trough_format[ell]*this->dtheta_trough_Legendres_dtheta_trough[ell];
    this->current_P_delta_NL_in_trough_format[ell] = Tell*this->current_P_NL_at(ln_k);
    this->current_P_delta_NL_in_trough_format_times_trough_legendres[ell] = this->current_P_delta_NL_in_trough_format[ell]*this->theta_trough_Legendres[ell];
  }
}


void Matter::set_linear_Legendres(int modus, double theta){
  
  double cos_th = cos(theta);
  double *Pl_th = new double[this->ell_max+2]; 
  gsl_sf_legendre_Pl_array(this->ell_max+1, cos_th, Pl_th);
  double A;
  give_area_from_cosines(1.0, cos_th, &A);
  
  if(modus == 1){
    for(int ell = 1; ell < this->ell_max; ell++){
      this->current_theta1_linear_Legendres[ell] = (Pl_th[ell+1]-Pl_th[ell-1])*constants::pi/((2.0*double(ell)+1.0)*A);
    }
  }
  
  if(modus == 2){
    double one_over_A = 1.0/A;
    for(int ell = 1; ell < this->ell_max; ell++){
      this->current_theta2_linear_Legendres[ell] = (Pl_th[ell+1]-Pl_th[ell-1])*one_over_A;
    }
  }
  
  if(modus == 3){
    double sin_th = sin(theta);
    double one_over_A = 1.0/A;
    double minus_sin_th_over_A = -sin_th/A;
    double pi2 = 2.0*constants::pi;
    for(int ell = 1; ell < this->ell_max; ell++){
      this->current_theta2_linear_Legendres[ell] = (Pl_th[ell+1]-Pl_th[ell-1])*one_over_A;
      this->current_theta2_linear_Legendres_prime[ell] = (this->current_theta2_linear_Legendres[ell]*pi2 + Pl_th[ell]*(2.0*double(ell)+1.0))*minus_sin_th_over_A;
    }
  }
  
  if(modus == 4){
    double sin_th = sin(theta);
    double one_over_A = 1.0/A;
    double minus_sin_th_over_A = -sin_th/A;
    double pi2 = 2.0*constants::pi;
    for(int ell = 1; ell < this->ell_max; ell++){
      this->current_theta1_linear_Legendres[ell] = (Pl_th[ell+1]-Pl_th[ell-1])*one_over_A*constants::pi/(2.0*double(ell)+1.0);
      this->current_theta1_linear_Legendres_prime[ell] = (this->current_theta1_linear_Legendres[ell]*pi2 + Pl_th[ell]*constants::pi)*minus_sin_th_over_A;
    }
  }
  
  /*if(modus == 4){
    double sin_th = sin(theta);
    for(int ell = 1; ell < this->ell_max; ell++){
      this->current_theta1_linear_Legendres[ell] = (Pl_th[ell+1]-Pl_th[ell-1])/A*constants::pi/(2.0*double(ell)+1.0);
      this->current_theta1_linear_Legendres_prime[ell] = (this->current_theta1_linear_Legendres[ell]/(1.0-cos_th) + Pl_th[ell]*(2.0*double(ell)+1.0)/A*constants::pi/((2.0*double(ell)+1.0)));
      this->current_theta1_linear_Legendres_prime[ell] *= -sin_th;
    }
  }*/

  delete []Pl_th;
  
}


void Matter::update_linear_power_spectrum(BINNING binning, vector<double> k_values, vector<double> P_k_lin_values){
  
  double k;
  
  vector<double> interpolation_support(k_values.size(),0.0);
  vector<double> new_transferfunction(k_values.size(),0.0);
  
  if(binning == LIN){
    interpolation_support = k_values;
  }
  else if(binning == LOG){
    for(int i = 0; i < k_values.size(); i++){
      interpolation_support[i] = log(k_values[i]);
    }
  }
  else{
    cerr << "WRONG VALUE OF ENUM BINNING IN update_linear_power_spectrum!" << '\n';
    cerr << "Abbort." << '\n';
    exit(1);
  }
  
  if(k_values.size() != P_k_lin_values.size()){
    cerr << "SIZES OF k_values AND P_k_lin_values DONT MATCH IN update_linear_power_spectrum!" << '\n';
    cerr << "Abbort." << '\n';
    exit(1);
  }
  
  for(int i = 0; i < k_values.size(); i++){
    k = k_values[i];
    new_transferfunction[i] = sqrt(P_k_lin_values[i]/(pow(k, this->cosmology.n_s)));
  }
  for(int i = 0; i < k_values.size(); i++){
    new_transferfunction[i] /= new_transferfunction[0];
  }
  
  for(int i = 0; i < this->wave_numbers.size(); i++){
    k = this->wave_numbers[i]/c_over_e5;
    if(binning == LOG)
      this->transfer_function[i] = interpolate_neville_aitken(log(k), &interpolation_support, &new_transferfunction, this->order);
    if(binning == LIN)
      this->transfer_function[i] = interpolate_neville_aitken(k, &interpolation_support, &new_transferfunction, this->order);
  }
  
  
  this->norm = this->variance_of_matter_within_R(8.0);

  
}






