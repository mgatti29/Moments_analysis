#include <iostream>
#include <fstream>
#include <ctime>
//#include <armadillo>

extern "C" double print_Cls(double Omega_m, double Omega_b, double sigma_8, double n_s, double h_100, char **l_file1, int number_of_bins,  long* bins2, int len_bins2,long* bins3, int len_bins3, const char **save_out , int lensing_A, int lensing_B,double A0,double z0,double alpha0,double* z_shift,double* ssss){


  double theta_27 = 1.0094444444444444; //Planck theta_27
  double a_min = 0.00025;
  double a_max = 1.001;
  double t_min = 2.0e-5;

  cosmological_model cosmo;

  cosmo.Omega_m = Omega_m;
  cosmo.Omega_r = 8.469833648509143e-05; //Planck Omega_r
  cosmo.Omega_L = 1.0 - cosmo.Omega_m - cosmo.Omega_r;
  cosmo.Omega_b = Omega_b;
  cosmo.Omega_k = 0.0;

  cosmo.n_s = n_s;
  cosmo.h_100 = h_100;
  cosmo.theta_27 = theta_27;
  cosmo.sigma_8 = sigma_8;
  cosmo.w0 = -1.0;
  cosmo.w1 = 0.0;

  Universe our_universe(cosmo, t_min, a_min, a_max, 1);



  Matter our_matter(&our_universe, 0.0);


  //*****************************************

  Matter_2D our_projection(&our_matter, l_file1, number_of_bins,z_shift,ssss);

  our_projection.set_cl_compute(1); //set smoothing to 0; defined in
  our_projection.matter->set_permanent_Legendres_null(constants::ell_max, 0.000); //set smoothing to 0

  our_projection.set_sd(lensing_A,lensing_B); //defined in Matter_2D.h
  our_projection.set_IA_params( A0,z0,alpha0); //defined in Matter_2D.h


  our_projection.set_bins(bins2,bins3,len_bins2,len_bins3); //defined in Matter_2D.h
  our_projection.set_output(save_out); //defined in Matter_2D.h


  //USE THIS FOR CLSSS (but we also hve to set sm = 0 in matter_2D_PDF_computation)

  our_projection.compute_Cl_kappa_total_lens();
  our_matter.print_power_spectra(.0);
  //*****************************************
   FILE* F = fopen("./pk.txt", "w");
   fclose(F);
   fstream out;
   out.open("./pk.txt");
   for(int ii=0; ii<1000;ii++){
       double k = ii*0.015-4.;
       double PL = our_matter.transfer_function_at(exp(k));
       double PL1 = our_matter.Newtonian_linear_power_spectrum(exp(k));
       double PNL = our_matter.current_P_NL_at(k);
       out<<k<<"    "<<PNL<<endl;
   }
   out.close();
  //our_projection.print_2D_stats_lens(save_out, 1.0); //routine in matter_2D.h



  return  0.;

}

extern "C" double print_moments(const char* nn, double* theta2, int size_theta, double Omega_m, double Omega_b, double sigma_8, double n_s, double h_100,char ** l_file1, int number_of_bins,  long* bins2, int len_bins2,long* bins3, int len_bins3, const char **save_out , int len_mask, const char *name_CL_pix_mask,int len_pix_mask, double fact_area, int lensing_A, int lensing_B,const char *datan, const char *inv_covn,double A0,double z0,double alpha0,double* z_shift,double* ssss, double* mask, char ** outp_d23,int OWL,double r_OWL,const char* DM_FILENAMEL,const char* U_FILENAME,const char* L_FILENAME,const char* powz, const char*powk, const char*powele, int NL_p){
  //cout<<size_theta<<" "<<  Omega_b<<" "<<  sigma_8<<" "<<  n_s<<" "<< h_100 << mask_E<<" "<< mask_B<<" "<<  len_mask<<" "<< name_CL_pix_mask<<" "<<  len_pix_mask<<endl;
  // cout<<" \n"<<  Omega_b<<" "<<  sigma_8<<" "<<  n_s<<" "<< h_100 << " " <<datan <<" " <<inv_covn <<endl;

  double theta_27 = 1.0094444444444444; //Planck theta_27
  double a_min = 0.00025;
  double a_max = 1.001;
  double t_min = 2.0e-5;


  vector<vector<double>> variance_values_output(size_theta, vector<double> (len_bins2, 0.0));
  vector<vector<double>> skewness_values_output(size_theta, vector<double> (len_bins3, 0.0));
  vector<double> theory(size_theta*(len_bins2+len_bins3), 0.0);
  cosmological_model cosmo;

  cosmo.Omega_m = Omega_m;
  cosmo.Omega_r = 8.469833648509143e-05; //Planck Omega_r
  cosmo.Omega_L = 1.0 - cosmo.Omega_m - cosmo.Omega_r;
  cosmo.Omega_b = Omega_b;
  cosmo.Omega_k = 0.0;

  cosmo.n_s = n_s;
  cosmo.h_100 = h_100;
  cosmo.theta_27 = theta_27;
  cosmo.sigma_8 = sigma_8;
  cosmo.w0 = -1.0;
  cosmo.w1 = 0.0;


  double theta1 [size_theta];
  double theta [size_theta];


  fstream inE;
  //inE.open("./angles.txt");
  inE.open(nn);
  double dummy;

  int j = 0;

  while(inE.good()){
    inE >> dummy;
    theta[j] = dummy;
    j = j+1;
    }
  inE.close();



  for (int sm = 0; sm< size_theta; sm++){
    theta1[sm] = theta[sm];
    theta[sm] = theta[sm]*constants::arcmin;
    
  }

  Universe our_universe(cosmo, t_min, a_min, a_max, 1);


  Matter our_matter(&our_universe, 0.0);
  our_matter.print_power_spectra(.5);
  our_matter.load_masks(mask,len_mask,size_theta,fact_area); //defined in Matter.h
  our_matter.read_pix_func(name_CL_pix_mask,len_pix_mask); //defined in Matter.h
  our_matter.set_d23_output(outp_d23);
  our_matter.set_NL_p(NL_p);
  our_matter.set_OWL(OWL,r_OWL,DM_FILENAMEL,U_FILENAME,L_FILENAME,powz,powk,powele);


  //our_matter.set_masks(mask_E,mask_B,len_mask,fact_area,name_CL_pix_mask,len_pix_mask);

//double A0,double z0,double eta,double z_shift
//  cout <<z_shift[0]<<"  "<<ssss[0]<<endl;
  Matter_2D our_projection(&our_matter, l_file1, number_of_bins,z_shift,ssss); //define in Matter_2D.h

    
    
  our_projection.set_cl_compute(0);
    
// the following commented code saves Cl.txt
  //our_projection.set_cl_compute(1); //set smoothing to 0; defined in
  //our_projection.matter->set_permanent_Legendres_null(constants::ell_max, 0.000); //set smoothing to 0
  //our_projection.set_sd(lensing_A,lensing_B); //defined in Matter_2D.h
  //our_projection.set_IA_params( A0,z0,alpha0); //defined in Matter_2D.h
  //our_projection.set_bins(bins2,bins3,len_bins2,len_bins3); //defined in Matter_2D.h
  //our_projection.set_output(save_out); //defined in Matter_2D.h
  //our_projection.compute_Cl_kappa_total_lens();
  //our_matter.print_power_spectra(.0);
    
    
    
    
  our_projection.set_sd(lensing_A,lensing_B); //defined in Matter_2D.h

  our_projection.set_IA_params( A0,z0,alpha0); //defined in Matter_2D.h
  vector<double> trough_variance;
  vector<double> trough_skewness;




  //*********
  //defined in matter_2D.h, and in matter_2D_pdf_computation.h//



  cout << "theta" << setw(12);
  cout << scientific << setprecision(5);
  for (int jj =0; jj< len_bins2; jj++){

        cout << bins2[2*jj] << ","<< bins2[2*jj+1]<< setw(13);
  }
  for (int jj =0; jj< len_bins3; jj++){
        cout << bins3[3*jj] << ","<< bins3[3*jj+1]<< ","<< bins3[3*jj+2]<< setw(11);
  }
  cout<<"\n";



   our_projection.compute_moments(theta, size_theta, &variance_values_output, &skewness_values_output,bins2,len_bins2,bins3,len_bins3);

   cout << setprecision(2);
   for(int sm = 0; sm< size_theta; sm++){
    cout << theta1[sm] << setw(15);
    cout << scientific << setprecision(5);
    for (int jj =0; jj< len_bins2; jj++){

        cout << variance_values_output[sm][jj] << setw(15);
    }


    for (int jj =0; jj< len_bins3; jj++){

        cout <<  skewness_values_output[sm][jj] << setw(15);
    }
        cout <<"\n";
  }




   //save file *************************************************

   int count = 0;
    //saving output by output and total *************



  for (int jj =0; jj< len_bins2; jj++){

    remove(save_out[count]);
    FILE* F = fopen(save_out[count], "w");
    fclose(F);
    fstream out;
    out.open(save_out[count]);

    for(int sm = 0; sm <size_theta; sm++){
        out << setprecision(2);
        out << theta1[sm] << setw(15);
        out << scientific << setprecision(5);
        out << variance_values_output[sm][jj] << setw(15);
        out <<"\n";

      }
    count +=1;
    out.close();
    }

  for (int jj =0; jj< len_bins3; jj++){
    remove(save_out[count]);
    FILE* F = fopen(save_out[count], "w");
    fclose(F);
    fstream out;
    out.open(save_out[count]);

    for(int sm = 0; sm <size_theta; sm++){
        out << setprecision(2);
        out << theta1[sm] << setw(15);
        out << scientific << setprecision(5);
        out << skewness_values_output[sm][jj] << setw(15);
        out <<"\n";

      }
    count +=1;
    out.close();
    }




  return 0.;

}

