
class ProjectedField {

public:
  ProjectedField(string config_file, Universe *u, double Omega, double dz, double z_shift, double spread);
  ProjectedField(string config_file, Universe *u, double Omega, double dz);
  ProjectedField(string config_file, Universe *u, double Omega, double dz, double z_cut);
  ProjectedField(string config_file, Universe *u, double Omega, double dz, double z_cut, int no_shear_overload);
  ~ProjectedField();
  
  double weight_at_comoving_distance(double w);
  double nz_at_comoving_distance(double w);
  double norm_of_quantile(double z_quantile);
  int shear_or_delta;
  int source_at_single_z;
  int constant_comoving_density;
  int histogram_modus;


  void return_w_boundaries_and_pofw(vector<double> *boundaries, vector<double> *p);
  double return_w_max();
  double return_w_min();
  double z_to_w(double z);
  double w_to_z(double w);


private:
  
  int no_shear_overload;
  
  double Omega_m;
  double w_min_constant_comoving_density;
  double w_max_constant_comoving_density;
  double norm_constant_comoving_density;
  double source_redshift;
  double source_comoving_distance;
  double z_mean;
  double z_min;
  double z_max;
  double z_cut;
  double w_min;
  double w_max;
  double eta_maximal;
  double z_shift;
  vector<double> w_centers;
  vector<double> w_boundaries;
  vector<double> w_steps;
  vector<double> w_weights;
  vector<double> n_of_w;
  vector<double> weights;


  Universe *uni;
  
  void configure(string config_file);
  void configure(string config_file, double  z_shift,double  z_spread);

  void read_histogram(string nofz_file);
  void read_histogram(string nofz_file, double z_shift,double  z_spread); //marco was here
  void set_lensing_kernel();
  

};
