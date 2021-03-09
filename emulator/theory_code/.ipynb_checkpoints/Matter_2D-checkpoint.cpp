#include <gsl/gsl_randist.h>
#include "Matter_2D.h"
#include "Matter_2D_integrands.h"
#include "delta0_and_kappa0.h"
#include "Matter_2D_PDF_computation.h"
#include "Matter_2D_PDF_analysis.h"

/*******************************************************************************************************************************************************
 * 1.1
 * Description:
 *
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

Matter_2D::Matter_2D(Matter* content, string lens_file){

  this->matter = content;
  this->universe = this->matter->universe;
  this->cosmology = this->universe->return_cosmology();
  this->lens_kernels.push_back(ProjectedField(lens_file, this->universe, this->cosmology.Omega_m, 0.0));
  double z_lens_max = this->lens_kernels[0].w_to_z(this->lens_kernels[0].return_w_max());
  this->order = 4;

  this->w_max = this->lens_kernels[0].return_w_max();
  this->w_min = 0.0*this->lens_kernels[0].return_w_min();
  this->eta_max = this->universe->eta_at_a(1.0);
  this->eta_min = this->eta_max - this->w_max;
}

//Marco was here

Matter_2D::Matter_2D(Matter* content, char** lens_file, int number_of_bins, double* z_shift, double* z_spread){

  this->matter = content;
  this->z_shift = z_shift;
  this->z_spread = z_spread;
  this->universe = this->matter->universe;
  this->cosmology = this->universe->return_cosmology();

  for (int i =0; i< number_of_bins; i++){
    this->lens_kernels.push_back(ProjectedField(lens_file[i], this->universe, this->cosmology.Omega_m, 0.0,this->z_shift[i],this->z_spread[i]));
  }

  double z_lens_max = this->lens_kernels[0].w_to_z(this->lens_kernels[0].return_w_max());
  this->order = 4;

  this->w_max = this->lens_kernels[0].return_w_max();
  this->w_min = 0.0*this->lens_kernels[0].return_w_min();
  this->eta_max = this->universe->eta_at_a(1.0);
  this->eta_min = this->eta_max - this->w_max;
}

Matter_2D::Matter_2D(Matter* content, char** lens_file, int number_of_bins){

  this->matter = content;
  this->universe = this->matter->universe;
  this->cosmology = this->universe->return_cosmology();

  for (int i =0; i< number_of_bins; i++){

    this->lens_kernels.push_back(ProjectedField(lens_file[i], this->universe, this->cosmology.Omega_m, 0.0));
  }

  double z_lens_max = this->lens_kernels[0].w_to_z(this->lens_kernels[0].return_w_max());
  this->order = 4;

  this->w_max = this->lens_kernels[0].return_w_max();
  this->w_min = 0.0*this->lens_kernels[0].return_w_min();
  this->eta_max = this->universe->eta_at_a(1.0);
  this->eta_min = this->eta_max - this->w_max;
}


//Marco was here
Matter_2D::Matter_2D(Matter* content, string lens_file1,string lens_file2){

  this->matter = content;
  this->universe = this->matter->universe;
  this->cosmology = this->universe->return_cosmology();
  this->lens_kernels.push_back(ProjectedField(lens_file1, this->universe, this->cosmology.Omega_m, 0.0));
  this->lens_kernels_2.push_back(ProjectedField(lens_file2, this->universe, this->cosmology.Omega_m, 0.0));

  //As long as the z_array are the same it does not matter
  double z_lens_max = this->lens_kernels[0].w_to_z(this->lens_kernels[0].return_w_max());
  //double z_lens_max_1 = this->lens_kernels[0].w_to_z(this->lens_kernels[0].return_w_max());
  this->order = 4;

  this->w_max = this->lens_kernels[0].return_w_max();
  this->w_min = 0.0*this->lens_kernels[0].return_w_min();
  this->eta_max = this->universe->eta_at_a(1.0);
  this->eta_min = this->eta_max - this->w_max;
}


Matter_2D::Matter_2D(Matter* content, string lens_file, string source_file, double dz_lens, double dz_source, vector<double> source_biases){ 
  
  this->source_biases = source_biases;
  if(source_biases.size() < 1){
    cerr << "No source bias value supplied in constructor of Matter_2D.\n";
    exit(1);
  }
  
  this->matter = content;
  this->universe = this->matter->universe;
  this->cosmology = this->universe->return_cosmology();
  this->lens_kernels.push_back(ProjectedField(lens_file, this->universe, this->cosmology.Omega_m, dz_lens));
  double z_lens_max = 10.0;//this->lens_kernels[0].w_to_z(this->lens_kernels[0].return_w_max());
  this->source_kernels.push_back(ProjectedField(source_file, this->universe, this->cosmology.Omega_m, dz_source));
  this->source_kernels_overlap.push_back(ProjectedField(source_file, this->universe, this->cosmology.Omega_m, dz_source, z_lens_max));
  this->source_kernels_density.push_back(ProjectedField(source_file, this->universe, this->cosmology.Omega_m, dz_source, z_lens_max, 1));
  this->order = 4;
    
  this->w_max = this->lens_kernels[0].return_w_max();
  this->w_min = this->lens_kernels[0].return_w_min();
  this->eta_max = this->universe->eta_at_a(1.0);
  this->eta_min = this->eta_max - this->w_max;
  
  //30 time-steps to interpolate non-linear evolution:
  //this->matter->set_spherical_collapse_evolution_of_delta(this->lens_kernels[0].w_to_z(this->w_min)+dz_lens, this->lens_kernels[0].w_to_z(this->w_max)+dz_lens, 30);
  
}

Matter_2D::Matter_2D(Matter* content, string lens_file, vector<string> source_files, double dz_lens, vector<double> dz_source, vector<double> source_biases){ 
  
  this->source_biases = source_biases;
  if(source_biases.size() < source_files.size()){
    cerr << "Too few source bias values supplied in constructor of Matter_2D.\n";
    exit(1);
  }
  
  this->matter = content;
  this->universe = this->matter->universe;
  this->cosmology = this->universe->return_cosmology();
  this->lens_kernels.push_back(ProjectedField(lens_file, this->universe, this->cosmology.Omega_m, dz_lens));
  double z_lens_max = 10.0;//this->lens_kernels[0].w_to_z(this->lens_kernels[0].return_w_max());
  for(int i = 0; i < source_files.size(); i++){
    this->source_kernels.push_back(ProjectedField(source_files[i], this->universe, this->cosmology.Omega_m, dz_source[i]));
    this->source_kernels_overlap.push_back(ProjectedField(source_files[i], this->universe, this->cosmology.Omega_m, dz_source[i], z_lens_max));
    this->source_kernels_density.push_back(ProjectedField(source_files[i], this->universe, this->cosmology.Omega_m, dz_source[i], z_lens_max, 1));
  }
  this->order = 4;
    
  this->w_max = this->lens_kernels[0].return_w_max();
  this->w_min = this->lens_kernels[0].return_w_min();
  this->eta_max = this->universe->eta_at_a(1.0);
  this->eta_min = this->eta_max - this->w_max;
  
  //30 time-steps to interpolate non-linear evolution:
  this->matter->set_spherical_collapse_evolution_of_delta(this->lens_kernels[0].w_to_z(this->w_min)+dz_lens, this->lens_kernels[0].w_to_z(this->w_max)+dz_lens, 30);
  
}


/*******************************************************************************************************************************************************
 * 1.2 Destructor
 * Description:
 * 
 * Arguments:
 * 
 * 
*******************************************************************************************************************************************************/

Matter_2D::~Matter_2D(){
}

