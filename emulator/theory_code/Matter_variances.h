

/*******************************************************************************************************************************************************
 * 1.6 set_cylinder_variances
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

void Matter::set_cylinder_variances(){
  
  int n = this->wave_numbers.size()/8;
  
  this->log_top_hat_radii.resize(n, 0.0);
  this->top_hat_cylinder_variances.resize(n, 0.0);
  this->current_P_L = this->P_L(this->universe->eta_at_a(1.0));
  
  for(int i = 0; i < n; i++){
    this->log_top_hat_radii[i] = -log(wave_numbers[8*(n-1-i)]);
    this->top_hat_cylinder_variances[i] = this->variance_of_matter_within_R_2D(1.0/wave_numbers[8*(n-1-i)]);
  }
  
}


/*******************************************************************************************************************************************************
 * 3.3 variance_of_matter_within_R
 * Description:
 *  - computes variance of density field when smoothed with tophat filter. 
 * Arguments:
 *  - R: radius of tophat filter in Mpc/h
 * 
*******************************************************************************************************************************************************/


double Matter::variance_of_matter_within_R(double R){
 
  double s_sq = 0;
  double prefactors;
  
  prefactors = 1.0/(2.0*constants::pi_sq);
  integration_parameters params;
  params.top_hat_radius = R;
  params.n_s = this->cosmology.n_s;
  params.pointer_to_Matter = this;
  integration_parameters * pointer_to_params = &params;
  
  //s_sq = int_gsl_integrate_medium_precision(norm_derivs_gsl,(void*)pointer_to_params,log(minimal_wave_number),log(maximal_wave_number),NULL,1000);
  s_sq = int_gsl_integrate_medium_precision(norm_derivs_gsl,(void*)pointer_to_params,log(minimal_wave_number*c_over_e5),log(maximal_wave_number*c_over_e5),NULL,len_ww);
  return prefactors*s_sq;
  
}


/*******************************************************************************************************************************************************
 * 3.5 variance_of_matter_within_R_2D
 * Description:
 *  - computes variance of density field when averaged over cylinders of radius R and length L = 1. 
 * Arguments:
 *  - R: radius of tophat filter in Mpc/h
 * 
*******************************************************************************************************************************************************/

double Matter::variance_of_matter_within_R_2D(double R){
 
  double s_sq = 0.0;
  double prefactors;
  
  prefactors = 1.0/(2.0*constants::pi*c_over_e5*c_over_e5*c_over_e5);
  integration_parameters params;
  params.top_hat_radius = R;
  params.n_s = this->cosmology.n_s;
  params.pointer_to_Matter = this;
  integration_parameters * pointer_to_params = &params;

  s_sq = int_gsl_integrate_medium_precision(norm_derivs_2D_gsl,(void*)pointer_to_params,log(minimal_wave_number*c_over_e5),log(maximal_wave_number*c_over_e5),NULL,len_ww);

  return prefactors*s_sq;
  
}



double Matter::variance_of_matter_within_R_2D(){
 
  
  double variance = 0.0;

  for(int ell = 1; ell < ell_max; ell++){
    variance += this->current_P_delta_L_in_trough_format[ell]*pow(this->theta_trough_Legendres[ell], 2);
  }

  return variance;
  
}

void Matter::variance_of_matter_within_R_2D_arr(int size_theta, vector<double> *variance_array){


  vector<double> dummy_array(size_theta, 0.0);
  double dummy = 0.0;
  int dummy_index = 0;

  // no mask, no pixel window ************************
  if (this->len_mask == 0){
    if (this->len_pix_mask == 0){
    for(int ell = 1; ell < ell_max; ell++){
        for(int i=0; i<size_theta; i++){

            (*variance_array)[i] += this->current_P_delta_L_in_trough_format[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2);
        }
    }
    }else{

    // no mask, yes pix window ***************************
    for(int ell = 1; ell <  this->len_pix_mask; ell++){
        for(int i=0; i<size_theta; i++){


            (*variance_array)[i] += this->current_P_delta_L_in_trough_format[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2)*Cl_mask[ell]*Cl_mask[ell];
    }
    }
    }
   }else{

   vector<double> convolved_vector(this->len_mask, 0.0);
   for(int ell = 1; ell < this->len_mask; ell++){
        dummy = this->Cl_mask[ell]*this->Cl_mask[ell]*this->current_P_delta_L_in_trough_format[ell];
        //cout<<dummy<<endl;
        for(int ell_1 = 1; ell_1 < this->len_mask; ell_1++){
            convolved_vector[ell_1] += dummy*this->maskdue[ell_1*this->len_mask+ell]; //this might need to be transposed
        }
   }
   for(int i=0; i<size_theta; i++){
        for(int ell = 1; ell < this->len_mask; ell++){
        (*variance_array)[i] +=convolved_vector[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2);
        }
   }
   }

}


void Matter::variance_of_matter_within_R_2D_NL_arr(int size_theta, vector<double> *variance_array){



  vector<double> dummy_array(size_theta, 0.0);

  double dummy = 0.0;
  int dummy_index = 0;



  if (this->len_mask == 0){
    if (this->len_pix_mask == 0){
    for(int ell = 1; ell < ell_max; ell++){
        for(int i=0; i<size_theta; i++){

            (*variance_array)[i] += this->current_P_delta_NL_in_trough_format[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2);
        }
    }
    }else{


    for(int ell = 1; ell <  this->len_pix_mask; ell++){
        for(int i=0; i<size_theta; i++){


            (*variance_array)[i] += this->current_P_delta_NL_in_trough_format[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2)*Cl_mask[ell]*Cl_mask[ell];
    }
    }
    }
   }else{






   vector<double> convolved_vector(this->len_mask, 0.0);
   for(int ell = 1; ell < this->len_mask; ell++){
        dummy = this->Cl_mask[ell]*this->Cl_mask[ell]*this->current_P_delta_NL_in_trough_format[ell];
        //cout<<dummy<<endl;
        for(int ell_1 = 1; ell_1 < this->len_mask; ell_1++){
            convolved_vector[ell_1] += dummy*this->maskdue[ell_1*this->len_mask+ell]; //this might need to be transposed
        }
   }
   for(int i=0; i<size_theta; i++){
        for(int ell = 1; ell < this->len_mask; ell++){
        (*variance_array)[i] +=convolved_vector[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2);
        }
   }
   // start = clock();


    /*

   for(int ell = 1; ell < this->len_mask; ell++){

       dummy_index = (this->len_mask)*ell;

       // OLD dummy = this->Cl_mask[ell]*this->Cl_mask[ell]*this->current_P_delta_NL_in_trough_format[ell]*(this->fact_area);
       for(int i=0; i<size_theta; i++){

            dummy_array[i] = dummy*pow(this->Atheta_trough_Legendres[i][ell], 2);
       }

       // convolution
       for(int ell_1 = 1 + dummy_index; ell_1 < (this->len_mask + dummy_index); ell_1++){
            for(int i=0; i<size_theta; i++){
                //OLD (*variance_array)[i] += dummy_array[i]*this->maskdue[i*this->len_mask*this->len_mask+ell_1];//this->maskE_armad.slice(i)(ell_1-(dummy_index),ell);//this->maskE[i][ell_1];
                (*variance_array)[i] += dummy_array[i]*this->maskdue[i*this->len_mask*this->len_mask+ell_1];//this->maskE_armad.slice(i)(ell_1-(dummy_index),ell);//this->maskE[i][ell_1];

            }

        }


    }
*/




   }


}


void Matter::variance_of_matter_within_R_2D_NL_arr_mod(int size_theta, vector<double> *variance_array_a,vector<double> *variance_array_b,vector<double> *variance_array_c, double w){

  // I have to define correction factors from GM12 or SC01:
  double a,b,c;
  double a1,a2,a3,a4,a5,a6,a7,a8,a9;
    
  //if you want the original PT values, need to set below a,b,c = 1.
    
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
  //ns =  -this->cosmology.n_s;
    
  // new
  ns = this->current_n_eff;
    
  double d1,ln_k1 ,ln_k1_1 ;

  s8 =  this->cosmology.sigma_8;
  double Dp = this->Dp;



  vector<double> dummy_array(size_theta, 0.0);

  double dummy = 0.0;
  int dummy_index = 0;

  double muted = 0;
  double q = 0;

  if (this->len_mask == 0){
    if (this->len_pix_mask == 0){
    for(int ell = 1; ell < ell_max; ell++){
        ln_k1 = log((ell+0.5)/(w)-(ell+0.5)/(w)/20.);
        ln_k1_1 = log((ell+0.5)/(w)+(ell+0.5)/(w)/20.);
        d1 = log((ell+0.5)/(w*4344.*0.7)*21./20.) - log((ell+0.5)/(w*4344.*0.7)*19./20.);
        ns = (log(current_P_L_at(ln_k1_1))-log(current_P_L_at(ln_k1)))/d1;
        if(w==0.){
        ns = 1.;
        }
        q= ell/l_nl;
        
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

            muted = this->current_P_delta_NL_in_trough_format[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2);
            
            //cout <<'a: ' a <<' b: '<<b<<' c: '<<' muted '<<muted<< endl;
            (*variance_array_a)[i] += muted * a ;
            (*variance_array_b)[i] += muted * b ;
            (*variance_array_c)[i] += muted * c ;
        }
    }
    }else{


    for(int ell = 1; ell <  this->len_pix_mask; ell++){
        q= ell/l_nl;
        ln_k1 = log((ell+0.5)/(w)-(ell+0.5)/(w)/20.);
        ln_k1_1 = log((ell+0.5)/(w)+(ell+0.5)/(w)/20.);
        d1 = log((ell+0.5)/(w*4344.*0.7)*21./20.) - log((ell+0.5)/(w*4344.*0.7)*19./20.);
        ns = (log(current_P_L_at(ln_k1_1))-log(current_P_L_at(ln_k1)))/d1;
        if(w==0.){
        ns = 1.;
        }
        //ns =  -this->cosmology.n_s;
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
            muted = this->current_P_delta_NL_in_trough_format[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2)*Cl_mask[ell]*Cl_mask[ell];
            (*variance_array_a)[i] += muted * a;
            (*variance_array_b)[i] += muted * b;
            (*variance_array_c)[i] += muted * c;
    }
    }
    }
   }else{






   vector<double> convolved_vector(this->len_mask, 0.0);
   for(int ell = 1; ell < this->len_mask; ell++){
        dummy = this->Cl_mask[ell]*this->Cl_mask[ell]*this->current_P_delta_NL_in_trough_format[ell];
        //cout<<dummy<<endl;
        for(int ell_1 = 1; ell_1 < this->len_mask; ell_1++){
            convolved_vector[ell_1] += dummy*this->maskdue[ell_1*this->len_mask+ell]; //this might need to be transposed
        }
   }
   for(int i=0; i<size_theta; i++){
        for(int ell = 1; ell < this->len_mask; ell++){
        ln_k1 = log((ell+0.5)/(w)-(ell+0.5)/(w)/20.);
        ln_k1_1 = log((ell+0.5)/(w)+(ell+0.5)/(w)/20.);
        d1 = log((ell+0.5)/(w*4344.*0.7)*21./20.) - log((ell+0.5)/(w*4344.*0.7)*19./20.);
        ns = (log(current_P_L_at(ln_k1_1))-log(current_P_L_at(ln_k1)))/d1;
        if(w==0.){
        ns = 1.;
        }
            q= ell/l_nl;
            
            if (this->NL_p == 0){
                a = 1.;
                b = 1.;
                c = 1.;
            }else{
                        
                a = (1. + pow(s8*Dp,a6)*pow((0.7*(4-pow(2,ns))/(1+pow(2,2*ns+1))),0.5)*pow(q*a1,ns+a2))/(1+pow(q*a1,ns+a2));
                b = (1. + 0.2*a3*(ns+3)*pow(q*a7,ns+a8+3))/(1+pow(q*a7,ns+a8+3.5));
                c = (1. + 4.5*a4/(1.5+pow(ns+3,4))*pow(q*a5,ns+3+a9))/(1+pow(q*a5,ns+3.5+a9));
            }
             muted = convolved_vector[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2);
            (*variance_array_a)[i] += muted * a;
            (*variance_array_b)[i] += muted * b;
            (*variance_array_c)[i] += muted * c;
        }
   }





   }


}


void Matter::variance_of_matter_within_R_2D_arr_mod(int size_theta, vector<double> *variance_array_a,vector<double> *variance_array_b,vector<double> *variance_array_c,double w){

  // I have to define correction factors from GM12 or SC01:
  double a,b,c;
  double a1,a2,a3,a4,a5,a6,a7,a8,a9;
  double d1,ln_k1 ,ln_k1_1;
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
  if (this->NL_p == 2){
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
  //ns =  -this->cosmology.n_s;
  ns = this->current_n_eff;
  s8 =  this->cosmology.sigma_8;
  double Dp = this->Dp;
  cout << l_nl<<"  "<<Dp <<"  "<< w<<endl;



  vector<double> dummy_array(size_theta, 0.0);

  double dummy = 0.0;
  int dummy_index = 0;

  double muted = 0;
  double q = 0;
  if (this->len_mask == 0){
    if (this->len_pix_mask == 0){
    for(int ell = 1; ell < ell_max; ell++){
        q= ell/l_nl;
        ln_k1 = log((ell+0.5)/(w)-(ell+0.5)/(w)/20.);
        ln_k1_1 = log((ell+0.5)/(w)+(ell+0.5)/(w)/20.);
        d1 = log((ell+0.5)/(w*4344.*0.7)*21./20.) - log((ell+0.5)/(w*4344.*0.7)*19./20.);
        ns = (log(current_P_L_at(ln_k1_1))-log(current_P_L_at(ln_k1)))/d1;
        if(w==0.){
        ns = 1.;
        }
        
        //ns =  -this->cosmology.n_s;
        if (this->NL_p == 0){
            a = 1;
            b = 1;
            c = 1;
        }else{
            a = (1. + pow(s8*Dp,a6)*pow((0.7*(4-pow(2,ns))/(1+pow(2,2*ns+1))),0.5)*pow(q*a1,ns+a2))/(1+pow(q*a1,ns+a2));
             b = (1. + 0.2*a3*(ns+3)*pow(q*a7,ns+a8+3))/(1+pow(q*a7,ns+a8+3.5));
             c = (1. + 4.5*a4/(1.5+pow(ns+3,4))*pow(q*a5,ns+3+a9))/(1+pow(q*a5,ns+3.5+a9));
        }


        for(int i=0; i<size_theta; i++){

            muted = this->current_P_delta_L_in_trough_format[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2);
            (*variance_array_a)[i] += muted * a;
            (*variance_array_b)[i] += muted * b ;
            (*variance_array_c)[i] += muted * c ;
        }
    }
    }else{


    for(int ell = 1; ell <  this->len_pix_mask; ell++){
        q= ell/l_nl;
        ln_k1 = log((ell+0.5)/(w)-(ell+0.5)/(w)/20.);
        ln_k1_1 = log((ell+0.5)/(w)+(ell+0.5)/(w)/20.);
        d1 = log((ell+0.5)/(w*4344.*0.7)*21./20.) - log((ell+0.5)/(w*4344.*0.7)*19./20.);
        ns = (log(current_P_L_at(ln_k1_1))-log(current_P_L_at(ln_k1)))/d1;
        if(w==0.){
        ns = 1.;
        }
        
        //ns =  -this->cosmology.n_s;
        if (this->NL_p == 0){
            a = 1;
            b = 1;
            c = 1;
        }else{
            a = (1. + pow(s8*Dp,a6)*pow((0.7*(4-pow(2,ns))/(1+pow(2,2*ns+1))),0.5)*pow(q*a1,ns+a2))/(1+pow(q*a1,ns+a2));
            b = (1. + 0.2*a3*(ns+3)*pow(q*a7,ns+a8+3))/(1+pow(q*a7,ns+a8+3.5));
            c = (1. + 4.5*a4/(1.5+pow(ns+3,4))*pow(q*a5,ns+3+a9))/(1+pow(q*a5,ns+3.5+a9));
        }
        for(int i=0; i<size_theta; i++){
            muted = this->current_P_delta_L_in_trough_format[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2)*Cl_mask[ell]*Cl_mask[ell];
            (*variance_array_a)[i] += muted * a;
            (*variance_array_b)[i] += muted * b;
            (*variance_array_c)[i] += muted * c;
    }
    }
    }
   }else{






   vector<double> convolved_vector(this->len_mask, 0.0);
   for(int ell = 1; ell < this->len_mask; ell++){
        dummy = this->Cl_mask[ell]*this->Cl_mask[ell]*this->current_P_delta_L_in_trough_format[ell];
        //cout<<dummy<<endl;
        for(int ell_1 = 1; ell_1 < this->len_mask; ell_1++){
            convolved_vector[ell_1] += dummy*this->maskdue[ell_1*this->len_mask+ell]; //this might need to be transposed
        }
   }
   for(int i=0; i<size_theta; i++){
        for(int ell = 1; ell < this->len_mask; ell++){
            q= ell/l_nl;
        ln_k1 = log((ell+0.5)/(w)-(ell+0.5)/(w)/20.);
        ln_k1_1 = log((ell+0.5)/(w)+(ell+0.5)/(w)/20.);
        d1 = log((ell+0.5)/(w*4344.*0.7)*21./20.) - log((ell+0.5)/(w*4344.*0.7)*19./20.);
        ns = (log(current_P_L_at(ln_k1_1))-log(current_P_L_at(ln_k1)))/d1;
        if(w==0.){
        ns = 1.;
        }
            
        if (this->NL_p == 0){
            a = 1;
            b = 1;
            c = 1;
        }else{
            a = (1. + pow(s8*Dp,a6)*pow((0.7*(4-pow(2,ns))/(1+pow(2,2*ns+1))),0.5)*pow(q*a1,ns+a2))/(1+pow(q*a1,ns+a2));
            b = (1. + 0.2*a3*(ns+3)*pow(q*a7,ns+a8+3))/(1+pow(q*a7,ns+a8+3.5));
            c = (1. + 4.5*a4/(1.5+pow(ns+3,4))*pow(q*a5,ns+3+a9))/(1+pow(q*a5,ns+3.5+a9));
        }
             muted = convolved_vector[ell]*pow(this->Atheta_trough_Legendres[i][ell], 2);
            (*variance_array_a)[i] += muted * a;
            (*variance_array_b)[i] += muted * b;
            (*variance_array_c)[i] += muted * c;
        }
   }





   }


}







double Matter::variance_of_matter_within_R_2D_NL(){

  double variance = 0.0;
  double dummy = 0.0;
  int dummy_index = 0;


 //ell_max

  if (this->len_mask == 0){
    if (this->len_pix_mask == 0){
    for(int ell = 1; ell < ell_max; ell++){
        variance += this->current_P_delta_NL_in_trough_format[ell]*pow(this->theta_trough_Legendres[ell], 2);
    }
    }else{


    for(int ell = 1; ell <  this->len_pix_mask; ell++){

        //cout << "boia " <<  this->current_P_delta_NL_in_trough_format[ell] <<" "<< this->theta_trough_Legendres[ell]<<" " << Cl_mask[ell] <<endl;
        variance += this->current_P_delta_NL_in_trough_format[ell]*pow(this->theta_trough_Legendres[ell], 2)*Cl_mask[ell]*Cl_mask[ell];
    }
    }
   }else{

   for(int ell = 1; ell < this->len_mask; ell++){

       dummy_index = (this->len_mask)*ell;
       dummy = pow(this->theta_trough_Legendres[ell], 2)*this->Cl_mask[ell]*this->Cl_mask[ell]*this->current_P_delta_NL_in_trough_format[ell];

       for(int ell_1 = 1 + dummy_index; ell_1 < (this->len_mask + dummy_index); ell_1++){
            variance += dummy;//*maskE[ell_1];
            cout<<"shouldn't run this"<<endl;

        }

    }

   }
  return variance;
  
}

double Matter::covariance_of_matter_within_R_2D(){

  double covariance = 0.0;

  for(int ell = 1; ell < ell_max; ell++){
    covariance += this->current_P_delta_L_in_trough_format[ell]*this->current_theta1_linear_Legendres[ell]*this->current_theta2_linear_Legendres[ell];
  }

  return covariance;
  
}

double Matter::covariance_of_matter_within_R_2D_NL(){

  double covariance = 0.0;

  for(int ell = 1; ell < ell_max; ell++){
    covariance += this->current_P_delta_NL_in_trough_format[ell]*this->current_theta1_linear_Legendres[ell]*this->current_theta2_linear_Legendres[ell];
  }

  return covariance;
  
}

double Matter::dcov1_at() const {

  double covariance = 0.0;

  for(int ell = 1; ell < ell_max; ell++){
    covariance += this->current_P_delta_L_in_trough_format[ell]*this->current_theta2_linear_Legendres[ell]*this->current_theta1_linear_Legendres_prime[ell];
  }
  return covariance;
  
}

double Matter::dcov2_at() const {

  double covariance = 0.0;

  for(int ell = 1; ell < ell_max; ell++){
  covariance += this->current_P_delta_L_in_trough_format[ell]*this->current_theta1_linear_Legendres[ell]*this->current_theta2_linear_Legendres_prime[ell];
  }

  return covariance;
}


double Matter::covariance_of_matter_within_R_2D_NL(int bin){

  double covariance = 0.0;

  for(int ell = 1; ell < ell_max; ell++){
    covariance += this->current_P_delta_NL_in_trough_format[ell]*this->bin_Legendres[bin][ell]*this->theta_trough_Legendres[ell];
  }

  return covariance;
  
}


