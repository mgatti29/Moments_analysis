

using namespace constants;



class Universe {

 public:

  Universe(cosmological_model cosmo, double t_start, double a_min, double a_max, int expand_or_collapse);
  ~Universe();
  
  int return_number_of_entries();
  
  double return_t(int i);
  double return_eta(int i);
  double return_a(int i);  
  double return_H(int i);
  double f_k(double w);
  
  vector<double> return_parameters();
  
  cosmological_model return_cosmology();
  
  void print_background_cosmology(string filename);
  
  double estimate_initial_step_size();
  double a_at_eta(double e);
  double H_at_eta(double e);
  double H_prime_at_eta(double e);  
  double eta_at_a(double a);
   
  double rho_m_of_a(double scale); // All in units of TODAYS critical density
  double rho_r_of_a(double scale);
  double rho_L_of_a(double scale);
  double w_L_of_a(double scale);

 private:

   int expansion_or_collapse;
   int number_of_entries;
   int order;
   
   double t_initial;
   double eta_initial;
   double a_initial;
   double a_final;
   double precision;

   vector<double> a;
   vector<double> H;
   vector<double> H_prime;
   vector<double> eta;
   vector<double> t;
   
   cosmological_model cosmology;
   
   void set_background_cosmology();
   void set_number_of_entries(int n_entries);
   
   double hubble_start(double a_start);
   double hubble_prime(double a_start);
   
};

