




void Matter_2D::compute_quantiles(int r_int, string data_file, vector<double> *delta_values, vector<double> *PDF){
	
	int n = delta_values->size();
	
	double normalization = 0.0;
	double delta_mean = 0.0;
	double variance = 0.0;
	double skewness = 0.0;
	double delta_min = (*delta_values)[0];
	double delta_max = (*delta_values)[n-1];
	double D_delta = (*delta_values)[1] - (*delta_values)[0];
	double dmin, dmax, dd, quantile;
	
	vector<double> boundaries(0, 0.0);
	vector<double> simdelta(200, 0.0);
	vector<double> simPDF(200, 0.0);
	
	string out_file;
	FILE *FF;
	fstream out;
	
	fstream in;
	in.open(data_file.c_str());
	for(int i = 0; i < 200; i++){
		in >> simdelta[i];
		in >> simPDF[i];
	}
	in.close();
	
	
	  
  for(int i = 0; i < n; i++){
    normalization += D_delta*(*PDF)[i];
    delta_mean += D_delta*(*PDF)[i]*(*delta_values)[i];
  }
  variance = 0.0;
  skewness = 0.0;
  for(int i = 0; i < n; i++){
    variance += D_delta*(*PDF)[i]*pow((*delta_values)[i]-delta_mean/normalization, 2.0);
    skewness += D_delta*(*PDF)[i]*pow((*delta_values)[i]-delta_mean/normalization, 3.0);
  }
  
  cout << "norm:" << setw(20);
  cout << normalization << '\n';
  cout << "mean:" << setw(20);
  cout << delta_mean/normalization << '\n';
  cout << "variance:" << setw(20);
  cout << variance/normalization << '\n';
  cout << "std. dev.:" << setw(20);
  cout << sqrt(variance/normalization) << '\n';
  cout << "skewness:" << setw(20);
  cout << skewness/normalization << '\n';
	
	dmin = delta_min;
	dmax = delta_min;
	dd = 0.000001;
	quantile = 0.0;
	boundaries.push_back(dmin);
	for(int i = 0; i < 10; i++){
		while(quantile < 0.1 && dmax < delta_max){
			quantile += dd*interpolate_Newton(dmax+0.5*dd, delta_values, PDF, 3);
			dmax += dd;
		}
		boundaries.push_back(dmax);
		quantile = 0.0;
	}
	
	cout << "\n\n\n";
	
	out_file = "quantiles_model_0.209966091999_0.445324094525_"+to_string(r_int)+".dat";
	remove(out_file.c_str());
	FF = fopen(out_file.c_str(), "w");
	fclose(FF);
	out.open(out_file.c_str());
	out << scientific << setprecision(10);
	
	for(int i = 0; i < 10; i++){
		normalization = 0.0;
		double delta_bar = 0.0;
		double d, p;
		dd = (boundaries[i+1] - boundaries[i])/99999.0;
		for(int j = 0; j < 100000; j++){
			d = boundaries[i] + (double(j)+0.5)*dd;
			p = interpolate_Newton(d, delta_values, PDF, 3);
			delta_bar += dd*d*p;
			normalization += dd*p;
		}
		cout << i << setw(20) << normalization << setw(20) << delta_bar/normalization << '\n';
		out << i << setw(20) << normalization << setw(20) << delta_bar/normalization << '\n';
	}
	out.close();
	
	
	boundaries = vector<double>(0, 0.0);
	dmin = simdelta[0];
	dmax = dmin;
	dd = 0.000001;
	quantile = 0.0;
	boundaries.push_back(dmin);
	for(int i = 0; i < 10; i++){
		while(quantile < 0.1 && dmax < simdelta[199]){
			quantile += dd*interpolate_Newton(dmax+0.5*dd, &simdelta, &simPDF, 3);
			dmax += dd;
		}
		boundaries.push_back(dmax);
		quantile = 0.0;
	}
	
	cout << "\n\n\n";
	
	
	out_file = "quantiles_sims_0.209966091999_0.445324094525_"+to_string(r_int)+".dat";
	remove(out_file.c_str());
	FF = fopen(out_file.c_str(), "w");
	fclose(FF);
	out.open(out_file.c_str());
	out << scientific << setprecision(10);
	
	
	for(int i = 0; i < 10; i++){
		normalization = 0.0;
		double delta_bar = 0.0;
		double d, p;
		dd = (boundaries[i+1] - boundaries[i])/99999.0;
		for(int j = 0; j < 100000; j++){
			d = boundaries[i] + (double(j)+0.5)*dd;
			p = interpolate_neville_aitken(d, &simdelta, &simPDF, 3);
			delta_bar += dd*d*p;
			normalization += dd*p;
		}
		cout << i << setw(20) << normalization << setw(20) << delta_bar/normalization << '\n';
		out << i << setw(20) << normalization << setw(20) << delta_bar/normalization << '\n';
	}
	out.close();
	
	boundaries = vector<double>(0, 0.0);
	dmin = delta_min;
	dmax = dmin;
	dd = 0.000001;
	quantile = 0.0;
	boundaries.push_back(dmin);
	for(int i = 0; i < 10; i++){
		while(quantile < 0.1 && dmax < simdelta[199]){
			quantile += dd*1.0/sqrt(2.0*constants::pi*variance)*exp(-0.5*(dmax+0.5*dd)*(dmax+0.5*dd)/variance);
			dmax += dd;
		}
		boundaries.push_back(dmax);
		quantile = 0.0;
	}
	
	cout << "\n\n\n";
		
	out_file = "quantiles_Gaussian_0.209966091999_0.445324094525_"+to_string(r_int)+".dat";
	remove(out_file.c_str());
	FF = fopen(out_file.c_str(), "w");
	fclose(FF);
	out.open(out_file.c_str());
	out << scientific << setprecision(10);
	
	
	for(int i = 0; i < 10; i++){
		normalization = 0.0;
		double delta_bar = 0.0;
		double d, p;
		dd = (boundaries[i+1] - boundaries[i])/9999.0;
		for(int j = 0; j < 10000; j++){
			d = boundaries[i] + (double(j)+0.5)*dd;
			p = 1.0/sqrt(2.0*constants::pi*variance)*exp(-0.5*d*d/variance);
			delta_bar += dd*d*p;
			normalization += dd*p;
		}
		cout << i << setw(20) << normalization << setw(20) << delta_bar/normalization << '\n';
		out << i << setw(20) << normalization << setw(20) << delta_bar/normalization << '\n';
	}
	out.close();
	
	
}










void Matter_2D::compute_moments(vector<double> *delta_values, vector<double> *PDF){
	
	int n = delta_values->size();
	
	double normalization = 0.0;
	double delta_mean = 0.0;
	double variance = 0.0;
	double skewness = 0.0;
	double delta_min = (*delta_values)[0];
	double delta_max = (*delta_values)[n-1];
	double D_delta = (*delta_values)[1] - (*delta_values)[0];
	double dmin, dmax, dd, quantile;
	
	
	  
  for(int i = 0; i < n; i++){
    normalization += D_delta*(*PDF)[i];
    delta_mean += D_delta*(*PDF)[i]*(*delta_values)[i];
  }
  variance = 0.0;
  skewness = 0.0;
  for(int i = 0; i < n; i++){
    variance += D_delta*(*PDF)[i]*pow((*delta_values)[i]-delta_mean/normalization, 2.0);
    skewness += D_delta*(*PDF)[i]*pow((*delta_values)[i]-delta_mean/normalization, 3.0);
  }
  
  cout << "norm:" << setw(20);
  cout << normalization << '\n';
  cout << "mean:" << setw(20);
  cout << delta_mean/normalization << '\n';
  cout << "variance:" << setw(20);
  cout << variance/normalization << '\n';
  cout << "std. dev.:" << setw(20);
  cout << sqrt(variance/normalization) << '\n';
  cout << "skewness:" << setw(20);
  cout << skewness/normalization << '\n';

}


void Matter_2D::compute_PofN(double n_bar, double bias, vector<double> *delta_values, vector<double> *PDF, vector<double> *PofN){
	
	int n = delta_values->size();
	int N = PofN->size();
	int fact = 1;
	
	double normalization = 0.0;
	double D_delta = (*delta_values)[1] - (*delta_values)[0];
	double d;
	
	
	  
  for(int i = 0; i < n; i++){
    normalization += D_delta*(*PDF)[i];
  }
  
  for(int i = 0; i < N; i++){
		(*PofN)[i] = 0.0;
    for(int j = 0; j < n; j++){
			d = bias*(*delta_values)[j];
			if(1.0+d > 0.0)
				(*PofN)[i] += D_delta*(*PDF)[j]*gsl_ran_poisson_pdf(i, n_bar*(1.0+d));
		}
		(*PofN)[i] /= normalization;
		
		fact *= i+1;
  }
	
	
} 
