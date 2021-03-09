
#include "ProjectedField.h"


ProjectedField::ProjectedField(string config_file, Universe *u,  double Omega,double dz, double z_shift,double spread){

  this->no_shear_overload = 0;
  this->z_cut = 10000.0;
  this->uni = u;
  this->Omega_m = Omega;
  this->z_shift = dz;
  this->eta_maximal = this->uni->eta_at_a(1.0);
  this->histogram_modus = 0;
  this->shear_or_delta = 0;
  this->source_at_single_z = 0;
  this->constant_comoving_density = 0;
  this->w_weights = vector<double>(len_ww, 0.0);
  this->weights = vector<double>(len_ww, 0.0);
  this->configure(config_file,z_shift,spread);

}


ProjectedField::ProjectedField(string config_file, Universe *u, double Omega, double dz){
  
  this->no_shear_overload = 0;
  this->z_cut = 10000.0;
  this->uni = u;
  this->Omega_m = Omega;
  this->z_shift = dz;
  this->eta_maximal = this->uni->eta_at_a(1.0);
  this->histogram_modus = 0;
  this->shear_or_delta = 0;
  this->source_at_single_z = 0;
  this->constant_comoving_density = 0;
  this->w_weights = vector<double>(len_ww, 0.0);
  this->weights = vector<double>(len_ww, 0.0);
  this->configure(config_file);

}

ProjectedField::ProjectedField(string config_file, Universe *u, double Omega, double dz, double z_cut){
  
  this->no_shear_overload = 0;
  this->z_cut = z_cut;
  this->uni = u;
  this->Omega_m = Omega;
  this->z_shift = dz;
  this->eta_maximal = this->uni->eta_at_a(1.0);
  this->histogram_modus = 0;
  this->shear_or_delta = 0;
  this->source_at_single_z = 0;
  this->constant_comoving_density = 0;
  this->w_weights = vector<double>(len_ww, 0.0);
  this->weights = vector<double>(len_ww, 0.0);
  this->configure(config_file);

}

ProjectedField::ProjectedField(string config_file, Universe *u, double Omega, double dz, double z_cut, int no_shear_overload){
  
  this->no_shear_overload = no_shear_overload;
  this->z_cut = z_cut;
  this->uni = u;
  this->Omega_m = Omega;
  this->z_shift = dz;
  this->eta_maximal = this->uni->eta_at_a(1.0);
  this->histogram_modus = 0;
  this->shear_or_delta = 0;
  this->source_at_single_z = 0;
  this->constant_comoving_density = 0;
  this->w_weights = vector<double>(len_ww, 0.0);
  this->weights = vector<double>(len_ww, 0.0);
  this->configure(config_file);

}


ProjectedField::~ProjectedField(){
}

//Marco was
void ProjectedField::configure(string config_file, double  z_shift,double  z_spread){
  string dummy;
  int PDF_modus;
  double z_min_constant_comoving_density;
  double z_max_constant_comoving_density;
  fstream in;

  in.open(config_file.c_str());
  in >> dummy; in >> PDF_modus;
  in >> dummy; in >> this->shear_or_delta;

    
    
  //PDF_modus = 1;
  //this->shear_or_delta = 1;
      
  if(this->no_shear_overload == 1){
    this->shear_or_delta = 0;
  }
  switch(PDF_modus){
    case 1:

      this->read_histogram(config_file, z_shift,z_spread);
      break;
    case 2:
      cerr << "redshift modus 2 now disabled";
      exit(1);
      break;
    case 3:
      cerr << "redshift modus 3 now disabled";
      exit(1);
      break;
    case 4:
      cerr << "redshift modus 4 now disabled";
      exit(1);
      break;
    case 5:
      this->source_at_single_z = 1;
      this->shear_or_delta = 1;
      this->constant_comoving_density = 0;
      in >> this->source_redshift;
      this->source_redshift += this->z_shift;
      this->z_mean = this->source_redshift;
      this->z_min = 0.01;
      this->z_max = this->source_redshift;
      this->w_min = this->z_to_w(this->z_min);
      this->w_max = this->z_to_w(this->z_max);
      break;
    case 6:
      this->constant_comoving_density = 1;
      this->source_at_single_z = 0;
      this->shear_or_delta = 0;
      in >> z_min_constant_comoving_density;
      in >> z_max_constant_comoving_density;
      z_min_constant_comoving_density += this->z_shift;
      z_max_constant_comoving_density += this->z_shift;
      this->z_min = z_min_constant_comoving_density;
      this->z_max = z_max_constant_comoving_density;
      this->w_min = this->z_to_w(this->z_min);
      this->w_max = this->z_to_w(this->z_max);
      break;
  }
  in.close();

  if(this->shear_or_delta == 1 && this->source_at_single_z == 0){
    double w, w_prime, dw_prime, f1, f2;
    int n = this->w_boundaries.size();
    double dw = this->w_boundaries[n-1]/(1.*len_ww);
    for(int i = 1; i <= len_ww; i++){
      w = double(i)*dw;
      this->w_weights[i-1] = w;

      for(int j = 0; j < n-1; j++){
				if(this->w_boundaries[j] >= w){
	 				w_prime = this->w_centers[j];
			  	f1 = this->uni->f_k(w_prime - w);
			  	f2 = this->uni->f_k(w_prime);
		  		this->weights[i-1] += this->w_steps[j]*this->n_of_w[j]*f1/f2;

				}
				else if(this->w_boundaries[j+1] >= w){
	 				w_prime = 0.5*(w+this->w_boundaries[j+1]);
	  			dw_prime = this->w_boundaries[j+1]-w;
	  			f1 = this->uni->f_k(w_prime - w);
	  			f2 = this->uni->f_k(w_prime);
	  			this->weights[i-1] += dw_prime*this->n_of_w[j]*f1/f2;
				}
      }
      this->weights[i-1] *= 1.5*this->Omega_m;
    }
  }


  if(this->source_at_single_z == 1){
    double a;
    a = 1.0/(1.0+this->source_redshift);
    this->source_comoving_distance = this->eta_maximal - this->uni->eta_at_a(a);
  }


  if(this->constant_comoving_density == 1){
    double a_min, a_max;
    a_max = 1.0/(1.0+this->z_min);
    a_min = 1.0/(1.0+this->z_max);
    this->norm_constant_comoving_density = (pow(this->w_max, 3.0) - pow(this->w_min, 3.0))/3.0;
  }
}





void ProjectedField::configure(string config_file){
  string dummy;
  int PDF_modus;
  double z_min_constant_comoving_density;
  double z_max_constant_comoving_density;
  fstream in;

  in.open(config_file.c_str());
  in >> dummy; in >> PDF_modus;
  in >> dummy; in >> this->shear_or_delta;

  if(this->no_shear_overload == 1){
    this->shear_or_delta = 0;
  }
  switch(PDF_modus){
    case 1:

      this->read_histogram(config_file);
      break;
    case 2: 
      cerr << "redshift modus 2 now disabled";
      exit(1);
      break;
    case 3: 
      cerr << "redshift modus 3 now disabled";
      exit(1);
      break;
    case 4: 
      cerr << "redshift modus 4 now disabled";
      exit(1);
      break;
    case 5: 
      this->source_at_single_z = 1;
      this->shear_or_delta = 1;
      this->constant_comoving_density = 0;
      in >> this->source_redshift;
      this->source_redshift += this->z_shift;
      this->z_mean = this->source_redshift;
      this->z_min = 0.01;
      this->z_max = this->source_redshift;
      this->w_min = this->z_to_w(this->z_min);
      this->w_max = this->z_to_w(this->z_max);
      break;
    case 6: 
      this->constant_comoving_density = 1;
      this->source_at_single_z = 0;
      this->shear_or_delta = 0;
      in >> z_min_constant_comoving_density;
      in >> z_max_constant_comoving_density;
      z_min_constant_comoving_density += this->z_shift;
      z_max_constant_comoving_density += this->z_shift;
      this->z_min = z_min_constant_comoving_density;
      this->z_max = z_max_constant_comoving_density;
      this->w_min = this->z_to_w(this->z_min);
      this->w_max = this->z_to_w(this->z_max);
      break;
  }
  in.close();
  
  if(this->shear_or_delta == 1 && this->source_at_single_z == 0){
    double w, w_prime, dw_prime, f1, f2;
    int n = this->w_boundaries.size();
    double dw = this->w_boundaries[n-1]/(1.*len_ww);
    for(int i = 1; i <= len_ww; i++){
      w = double(i)*dw;
      this->w_weights[i-1] = w;
      for(int j = 0; j < n-1; j++){
				if(this->w_boundaries[j] >= w){
	 				w_prime = this->w_centers[j];
			  	f1 = this->uni->f_k(w_prime - w);
			  	f2 = this->uni->f_k(w_prime);
		  		this->weights[i-1] += this->w_steps[j]*this->n_of_w[j]*f1/f2;
				}
				else if(this->w_boundaries[j+1] >= w){
	 				w_prime = 0.5*(w+this->w_boundaries[j+1]);
	  			dw_prime = this->w_boundaries[j+1]-w;
	  			f1 = this->uni->f_k(w_prime - w);
	  			f2 = this->uni->f_k(w_prime);
	  			this->weights[i-1] += dw_prime*this->n_of_w[j]*f1/f2;
				}
      }
      this->weights[i-1] *= 1.5*this->Omega_m;
    }
  }
  
  
  if(this->source_at_single_z == 1){
    double a;
    a = 1.0/(1.0+this->source_redshift);
    this->source_comoving_distance = this->eta_maximal - this->uni->eta_at_a(a);
  }
  
  
  if(this->constant_comoving_density == 1){
    double a_min, a_max;
    a_max = 1.0/(1.0+this->z_min);
    a_min = 1.0/(1.0+this->z_max);
    this->norm_constant_comoving_density = (pow(this->w_max, 3.0) - pow(this->w_min, 3.0))/3.0;
  }  
}


void ProjectedField::set_lensing_kernel(){
  double w, w_prime, dw_prime, f1, f2;
    int n = this->w_boundaries.size();
    double dw = this->w_boundaries[n-1]/(1.*len_ww);
    for(int i = 1; i <= len_ww; i++){
      w = double(i)*dw;
      this->w_weights[i-1] = w;
      for(int j = 0; j < n-1; j++){
				if(this->w_boundaries[j] >= w){
	 				w_prime = this->w_centers[j];
			  	f1 = this->uni->f_k(w_prime - w);
			  	f2 = this->uni->f_k(w_prime);
		  		this->weights[i-1] += this->w_steps[j]*this->n_of_w[j]*f1/f2;
				}
				else if(this->w_boundaries[j+1] >= w){
	 				w_prime = 0.5*(w+this->w_boundaries[j+1]);
	  			dw_prime = this->w_boundaries[j+1]-w;
	  			f1 = this->uni->f_k(w_prime - w);
	  			f2 = this->uni->f_k(w_prime);
	  			this->weights[i-1] += dw_prime*this->n_of_w[j]*f1/f2;
				}
      }
      this->weights[i-1] *= 1.5*this->Omega_m;
  }
}

double ProjectedField::weight_at_comoving_distance(double w){
  
  int index;
  double eta = this->uni->eta_at_a(1.0) - w;
  double scale = this->uni->a_at_eta(eta);
  double z0 = 0.4;
  double alpha = 0.0;
    
  if(this->constant_comoving_density==1){
  	if(w < this->w_min || w > this->w_max)
  	  return 0.0;
  	
    return w/this->norm_constant_comoving_density/pow(scale*(1.0 + z0), alpha);
	}
  
  if(this->source_at_single_z == 0 && w > this->w_boundaries[this->w_boundaries.size()-1])
    return 0.0;
    
  if(this->source_at_single_z == 1 && w > this->source_comoving_distance){
    return 0.0;
	}

  if(this->shear_or_delta == 0){
    if(w < this->w_boundaries[0] || w <= 0.0)
      return 0.0;
    index = find_index(w, &this->w_boundaries);
    if(index < this->n_of_w.size())
      return this->n_of_w[index]/w/pow(scale*(1.0 + z0), alpha);
    else
      return this->n_of_w[index-1]/w/pow(scale*(1.0 + z0), alpha);
    cout << "hahaha"; cin >> w;
  }
  else{
  	if(this->source_at_single_z == 1){
	  	double f1 = this->uni->f_k(this->source_comoving_distance - w);
	  	double f2 = this->uni->f_k(this->source_comoving_distance);

	  	return 1.5*this->Omega_m*f1/f2/scale;
		}


    index = find_index(w, &this->w_weights);

    return this->weights[index]/scale;
  }
  return 0.0;
}


//

double ProjectedField::nz_at_comoving_distance(double w){

  int index;
  double eta = this->uni->eta_at_a(1.0) - w;
  double scale = this->uni->a_at_eta(eta);
  double z0 = 0.4;
  double alpha = 0.0;

  if(this->constant_comoving_density==1){
  	if(w < this->w_min || w > this->w_max)
  	  return 0.0;

    return w/this->norm_constant_comoving_density/pow(scale*(1.0 + z0), alpha);
	}

  if(this->source_at_single_z == 0 && w > this->w_boundaries[this->w_boundaries.size()-1])
    return 0.0;

  if(this->source_at_single_z == 1 && w > this->source_comoving_distance){
    return 0.0;
	}


    if(w < this->w_boundaries[0] || w <= 0.0)
      return 0.0;
    index = find_index(w, &this->w_boundaries);
    if(index < this->n_of_w.size())
      return this->n_of_w[index]/w/pow(scale*(1.0 + z0), alpha);
    else
      return this->n_of_w[index-1]/w/pow(scale*(1.0 + z0), alpha);
    cout << "hahaha"; cin >> w;


  return 0.0;
}


//Marco was here
void ProjectedField::read_histogram(string nofz_file, double z_shift1,double z_spread1){
  string dummy_string;
  fstream in;
  in.open(nofz_file);

  getline(in, dummy_string);
  getline(in, dummy_string);

  this->histogram_modus = 1;
  double dummy, dummy_z = 0.0, normalisation = 0.0, scale, dz, dw;
  vector<double> z(0, 0.0);
  vector<double> N_of_z(0, 0.0);

  vector<double> z_mute(0, 0.0);
  vector<double> N_of_z_mute(0, 0.0);
  double weight_at_negative_z=0.;


  // read first just to ompute the mean and compute the spread
  while(in.good() && dummy_z < this->z_cut){
    in >> dummy_z;
    dummy_z += z_shift1;//this->z_shift; Marco here
    if(in.good() && dummy_z < this->z_cut){
      in >> dummy;

       z_mute.push_back(dummy_z);
       //cout<<dummy_z<<"  "<<dummy<<endl;
       N_of_z_mute.push_back(dummy);

    }
    else if(in.good()){
      z_mute.push_back(this->z_cut);
      in >> dummy;
      N_of_z_mute.push_back(0.0);
      N_of_z_mute[N_of_z_mute.size()-2] *= (this->z_cut - z_mute[z.size()-2])/(dummy_z - z_mute[z_mute.size()-2]);
    }
  }
  in.close();
  //compute the mean, and spread

  double mean = 0.;
  double norm_mean = 0.;
  for(int i = 0; i <  N_of_z_mute.size(); i++){
    mean +=  z_mute[i]*N_of_z_mute[i];
    norm_mean +=  N_of_z_mute[i];

  }
  mean = mean/norm_mean;
  cout<< "Shift: "<<z_shift1<<" Spread: "<< z_spread1<<endl;
  cout<< "Mean:  "<<mean-z_shift1<<endl;

  in.open(nofz_file);
  getline(in, dummy_string);
  getline(in, dummy_string);

  while(in.good() && dummy_z < this->z_cut){
    in >> dummy_z;
    dummy_z += z_shift1;//this->z_shift; Marco here
    dummy_z = (dummy_z - mean)*z_spread1+mean;

    if(in.good() && dummy_z < this->z_cut){
      in >> dummy;

      if(dummy_z<0) {
        weight_at_negative_z += dummy;
      } else {
        z.push_back(dummy_z);
        //cout<<dummy_z<<"  "<<dummy<<endl;
        N_of_z.push_back(dummy);
      }
    }
    else if(in.good()){
      z.push_back(this->z_cut);
      in >> dummy;

      N_of_z.push_back(0.0);
      N_of_z[N_of_z.size()-2] *= (this->z_cut - z[z.size()-2])/(dummy_z - z[z.size()-2]);
    }
  }
  in.close();



  int n = z.size();

  // catch cases where all or most sources are at negative z
  if(n==1) {
    z.push_back(z[0]+0.01);
    N_of_z.push_back(0.);
  }
  else if(n==0) {
    z.push_back(0.); z.push_back(0.01);
    N_of_z.push_back(1.);
    N_of_z.push_back(0.);
  }

  N_of_z[0] += weight_at_negative_z;

  mean=0;
  norm_mean=0;
  for(int i = 0; i < N_of_z.size(); i++){
    mean +=  z[i]*N_of_z[i];
    norm_mean +=  N_of_z[i];

  }
  mean = mean/norm_mean;
  cout<< "Mean after spread + shift:  "<<mean<<endl;



  this->w_boundaries = vector<double>(n, 0.0);
  this->w_centers = vector<double>(n-1, 0.0);
  this->w_steps = vector<double>(n-1, 0.0);
  this->n_of_w = vector<double>(n-1, 0.0);
  for(int i = 0; i < n; i++){
    scale = 1.0/(1.0+z[i]);
    this->w_boundaries[i] = this->eta_maximal - this->uni->eta_at_a(scale);
  }

  for(int i = 0; i < n-1; i++){
    dz = z[i+1]-z[i];
    this->w_centers[i] = this->eta_maximal - this->uni->eta_at_a(1.0/(1.0+z[i]+0.5*dz));

    dw = this->w_boundaries[i+1]-this->w_boundaries[i];
    this->n_of_w[i] = N_of_z[i]/dw;
    //cout<<this->n_of_w[i]<<" "<< N_of_z[i]<<endl;
    this->w_steps[i] = dw;
    normalisation += N_of_z[i];
  }
  for(int i = 0; i < n-1; i++){
    this->n_of_w[i] /= normalisation;
  }

  this->w_min = this->w_boundaries[0];
  this->w_max = this->w_boundaries[n-1];
  this->z_min = this->w_to_z(this->w_min);
  this->z_max = this->w_to_z(this->w_max);

}








void ProjectedField::read_histogram(string nofz_file){
  string dummy_string;
  fstream in;
  in.open(nofz_file);

  getline(in, dummy_string);
  getline(in, dummy_string);

  this->histogram_modus = 1;
  double dummy, dummy_z = 0.0, normalisation = 0.0, scale, dz, dw;
  vector<double> z(0, 0.0);
  vector<double> N_of_z(0, 0.0);

  double weight_at_negative_z=0.;

  while(in.good() && dummy_z < this->z_cut){
    in >> dummy_z;
    dummy_z += this->z_shift;
    if(in.good() && dummy_z < this->z_cut){
      in >> dummy;

      if(dummy_z<0) {
        weight_at_negative_z += dummy;
      } else {
        z.push_back(dummy_z);
        N_of_z.push_back(dummy);
      }
    }
    else if(in.good()){
      z.push_back(this->z_cut);
      in >> dummy;

      N_of_z.push_back(0.0);
      N_of_z[N_of_z.size()-2] *= (this->z_cut - z[z.size()-2])/(dummy_z - z[z.size()-2]);
    }
  }
  in.close();
  int n = z.size();

  // catch cases where all or most sources are at negative z
  if(n==1) {
    z.push_back(z[0]+0.01);
    N_of_z.push_back(0.); 
  }
  else if(n==0) {
    z.push_back(0.); z.push_back(0.01);
    N_of_z.push_back(1.); 
    N_of_z.push_back(0.); 
  }

  N_of_z[0] += weight_at_negative_z;
  
  this->w_boundaries = vector<double>(n, 0.0);
  this->w_centers = vector<double>(n-1, 0.0);
  this->w_steps = vector<double>(n-1, 0.0);
  this->n_of_w = vector<double>(n-1, 0.0);
  for(int i = 0; i < n; i++){
    scale = 1.0/(1.0+z[i]);
    this->w_boundaries[i] = this->eta_maximal - this->uni->eta_at_a(scale);
  }

  for(int i = 0; i < n-1; i++){
    dz = z[i+1]-z[i];
    this->w_centers[i] = this->eta_maximal - this->uni->eta_at_a(1.0/(1.0+z[i]+0.5*dz));
    dw = this->w_boundaries[i+1]-this->w_boundaries[i];
    this->n_of_w[i] = N_of_z[i]/dw;
    this->w_steps[i] = dw;
    normalisation += N_of_z[i];
  }
  for(int i = 0; i < n-1; i++){
    this->n_of_w[i] /= normalisation;
  }
  
  this->w_min = this->w_boundaries[0];
  this->w_max = this->w_boundaries[n-1];
  this->z_min = this->w_to_z(this->w_min);
  this->z_max = this->w_to_z(this->w_max);
  
}


double ProjectedField::norm_of_quantile(double z_quantile){
  
  if(this->shear_or_delta == 1){
    cerr << "Error in ProjectedField::norm_of_quantile: this method does not work in shear-mode!\n";
    exit(1);
  }
  
  double w_quantile = this->z_to_w(z_quantile);
  double normalisation = 0.0;
  double w = 0.0, dw;
  
  int n = this->w_steps.size(), i = 0;
  
  while(w <= w_quantile && i < n){
    dw = this->w_steps[i];
    w = this->w_boundaries[i+1];
    if(w <= w_quantile)
      normalisation += this->n_of_w[i]*dw;
    else
      normalisation += this->n_of_w[i]*(w_quantile - this->w_boundaries[i]);
    
    i++;
  }
  
  return normalisation;
  
}



void ProjectedField::return_w_boundaries_and_pofw(vector<double> *boundaries, vector<double> *p){
	
	int n = this->w_boundaries.size();
	
	(*p) = vector<double>(n-1, 0.0);
	(*boundaries) = vector<double>(n, 0.0);
	
	
  for(int i = 0; i < n-1; i++){
    (*p)[i] = this->n_of_w[i];
  }
  for(int i = 0; i < n; i++){
    (*boundaries)[i] = this->w_boundaries[i];
  }
	
}


double ProjectedField::z_to_w(double z){
  
  return this->uni->eta_at_a(1.0) - this->uni->eta_at_a(1.0/(1.0+z));
  
}

double ProjectedField::w_to_z(double w){
  
  return 1.0/this->uni->a_at_eta(this->uni->eta_at_a(1.0) - w) - 1.0;
  
}

double ProjectedField::return_w_max(){
  return this->w_max;
}

double ProjectedField::return_w_min(){
  return this->w_min;
}



