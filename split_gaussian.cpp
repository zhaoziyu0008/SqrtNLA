#include <iostream>
#include <math.h>
#include <NTL/LLL.h>
#include <tool.h>
#include <omp.h>

#define SEARCH_STEP 0.2
#define K_BOUND 3.0
#define S_BOUND -2.0
#define EPS 0.00001

using namespace NTL;
using namespace std;

inline double f(double x){
	return (1+erf(x/sqrt(2)))/2;
}
inline double g(double x){
	return (1-erf(x/sqrt(2)))/2;
}

struct split_res{
	double cost = 0.0;
	double sol = 1.0;
	double fal = 1.0;
};



void report_optimal_path_2(double goal){
	double fast =0.0;
	double fast_k1, fast_k2, fast_s1, fast_s2;
	double ccost, csol, cfalse;
	for (double k1 = SEARCH_STEP; k1 < K_BOUND; k1+= SEARCH_STEP){
		for (double s1 = -3.0; s1 < k1-0.00001; s1+=SEARCH_STEP){
			double cost1 = k1*k1;
			double sol1 = f(s1);
			double fal1 = g(k1-s1);
			for (double k2 = SEARCH_STEP; k2 < K_BOUND; k2+=SEARCH_STEP){
				for (double s2 = -3.0; s2 < k2-0.0001; s2+=SEARCH_STEP){
					double cost = cost1 + fal1 * k2 *k2;
					double fal = fal1 * g(k2-s2);
					double sol = sol1 * f(s2);
					if (sol/fal > goal){
						double speed = sol/cost;
						if (speed > fast){
							fast = speed;
							fast_k1 = k1;
							fast_k2 = k2;
							fast_s1 = s1;
							fast_s2 = s2;
							ccost = cost;
							csol = sol;
							cfalse = fal;
						}
					}
				}
			}
		}
	}
    printf("best k1 = %.2f, s1 = %.2f, k2 = %.2f, s2 = %.2f, cost = %.4f, sol = %.4f, false = %.4f\n", fast_k1, fast_s1, fast_k2, fast_s2, ccost, csol, cfalse);
}

void report_optimal_path_3(double goal){
	double fast = 0.0;
	double fk1, fk2, fk3, fs1, fs2, fs3;
	double ccost, csol, cfalse;
	pthread_spinlock_t min_lock = 1;
	omp_set_num_threads(12);
	#pragma omp parallel for
	for (long k1l = 1; k1l < (long)(K_BOUND/SEARCH_STEP); k1l++){
		double k1 = k1l * SEARCH_STEP;
		for (double s1 = S_BOUND; s1 < k1 - EPS; s1 += SEARCH_STEP){
			double cost1 = k1 * k1;
			double sol1 = f(s1);
			double fal1 = g(k1-s1);
			for (double k2 = SEARCH_STEP; k2 < K_BOUND; k2 += SEARCH_STEP){
				for (double s2 = S_BOUND; s2 < k2 - EPS; s2 += SEARCH_STEP){
					double cost2 = cost1 + fal1 * k2 * k2;
					double fal2 = fal1 * g(k2 - s2);
					double sol2 = sol1 * f(s2);
					for (double k3 = SEARCH_STEP; k3 < K_BOUND; k3 += SEARCH_STEP){
						for (double s3 = S_BOUND; s3 < k3 - EPS; s3 += SEARCH_STEP){
							double cost3 = cost2 + fal2 * k3 * k3;
							double fal3 = fal2 * g(k3 - s3);
							double sol3 = sol2 * f(s3);
							if (sol3/fal3 > goal){
								double speed = sol3 / cost3;
								if (speed > fast){
									pthread_spin_lock(&min_lock);
									fast = speed;
									fk1 = k1;
									fk2 = k2;
									fk3 = k3;
									fs1 = s1;
									fs2 = s2;
									fs3 = s3;
									ccost = cost3;
									csol = sol3;
									cfalse = fal3;
									pthread_spin_unlock(&min_lock);
								}
							}
						}
					}
				}
			}
		}
	}
	printf("best k1 = %.2f, s1 = %.2f, k2 = %.2f, s2 = %.2f, k3 = %.2f, s3 = %.2f, cost = %.4f, sol = %.4f, false = %.4f\n", fk1, fs1, fk2, fs2, fk3, fs3, ccost, csol, cfalse);

}

void report_optimal_path_4(double goal){
	double fast = 0.0;
	double fk1, fk2, fk3, fk4, fs1, fs2, fs3, fs4;
	double ccost, csol, cfalse;
	pthread_spinlock_t min_lock = 1;
	omp_set_num_threads(12);
	#pragma omp parallel for
	for (long k1l = 1; k1l < (long)(2/SEARCH_STEP); k1l++){
		double k1 = k1l * SEARCH_STEP;
		for (double s1 = -0.60; s1 < k1 - EPS; s1 += SEARCH_STEP){
			double cost1 = k1 * k1;
			double sol1 = f(s1);
			double fal1 = g(k1-s1);
			for (double k2 = SEARCH_STEP; k2 < 3.0; k2 += SEARCH_STEP){
				for (double s2 = -1.0; s2 < k2 - EPS; s2 += SEARCH_STEP){
					double cost2 = cost1 + fal1 * k2 * k2;
					double fal2 = fal1 * g(k2 - s2);
					double sol2 = sol1 * f(s2);
					for (double k3 = SEARCH_STEP; k3 < 5.0; k3 += SEARCH_STEP){
						for (double s3 = -1.0; s3 < k3 - EPS; s3 += SEARCH_STEP){
							double cost3 = cost2 + fal2 * k3 * k3;
							double fal3 = fal2 * g(k3 - s3);
							double sol3 = sol2 * f(s3);
							for (double k4 = 1.0; k4 < 7.0; k4 += SEARCH_STEP){
								for (double s4 = 0.0; s4 < k4-EPS; s4 += SEARCH_STEP){
									double cost4 = cost3 + fal3 * k4 * k4;
									double fal4 = fal3 * g(k4 - s4);
									double sol4 = sol3 * f(s4);
									if (sol4 / fal4 > goal){
										double speed = sol3 / cost3;
										if (speed > fast){
											pthread_spin_lock(&min_lock);
											fast = speed;
											fk1 = k1;
											fk2 = k2;
											fk3 = k3;
											fk4 = k4;
											fs1 = s1;
											fs2 = s2;
											fs3 = s3;
											fs4 = s4;
											ccost = cost4;
											csol = sol4;
											cfalse = fal4;
											pthread_spin_unlock(&min_lock);
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	printf("best k1 = %.2f, s1 = %.2f, k2 = %.2f, s2 = %.2f, k3 = %.2f, s3 = %.2f, k4 = %.2f, s4 = %.2f, cost = %.4f, sol = %.4f, false = %.4f\n", fk1, fs1, fk2, fs2, fk3, fs3, fk4, fs4, ccost, csol, cfalse);

}

split_res split(double k1, double s1, double k2, double s2, double k3, double s3, double k4, double s4){
	split_res ret;
	double cost = k1 * k1;
	double sol = f(s1);
	double fal = g(k1-s1);
	cost += k2 * k2 * fal;
	fal *= g(k2 - s2);
	sol *= f(s2);
	cost += k3 * k3 * fal;
	fal *= g(k3 - s3);
	sol *= f(s3);
	cost += k4 * k4 * fal;
	fal *= g(k4 - s4);
	sol *= f(s4);
	ret.fal = fal;
	ret.cost = cost;
	ret.sol = sol;
	return ret;
}

void random_walk_search_4(double goal){
	double k1, k2, k3, k4, s1, s2, s3, s4, cost, sol, fal;
	k1 = 0.2;
	s1 = 0.0;
	k2 = 0.2;
	s2 = 0.0;
	k3 = 0.2;
	s3 = 0.0;
	k4 = 3.8;
	s4 = 0.0;
	cost = split(k1, s1, k2, s2, k3, s3, k4, s4).cost;
	sol = split(k1, s1, k2, s2, k3, s3, k4, s4).sol;
	fal = split(k1, s1, k2, s2, k3, s3, k4, s4).fal;
	cout << cost << endl << sol << endl << fal << endl;
	uint64_t count = -1;
	while (true){
		count++;
		for (long i = 0; i < 1000000; i++){
			Vec<double> err = random_vec(8, 0.001);
			double err1 = err[0];
			double err2 = err[1];
			double err3 = err[2];
			double err4 = err[3];
			double err5 = err[4];
			double err6 = err[5];
			double err7 = err[6];
			double err8 = err[7];
			split_res res = split(k1+err1, s1+err2, k2+err3, s2+err4, k3+err5, s3+err6, k4+err7, s4+err8);
			if ((res.sol/res.fal > goal) && (res.sol / res.cost > sol / cost)){
				k1+=err1; 
				s1+=err2; 
				k2+=err3; 
				s2+=err4; 
				k3+=err5; 
				s3+=err6; 
				k4+=err7;
				s4+=err8;
				cost = res.cost; 
				sol = res.sol; 
				fal = res.fal;
			}
		}
		printf("[epoch %ld] k1 = %.3f, s1 = %.3f, k2 = %.3f, s2 = %.3f, k3 = %.3f, s3 = %.3f, k4 = %.3f, s4 = %.3f, cost = %.6f, sol = %.6f, false = %.6f\n", count, k1, s1, k2, s2, k3, s3, k4, s4, cost, sol, fal);
	}
}

split_res split(double *k, double *s, long n){
	split_res ret;
	for (long i = 0; i < n; i++){
		ret.cost += ret.fal * k[i] * k[i];
		if (i == 0) ret.cost *= 0.7;
		if (i == 1) ret.cost *= 0.7;
		if (i == n-1) ret.cost += ret.fal *k[i] * k[i];
		ret.fal *= g(k[i]-s[i]);
		ret.sol *= f(s[i]);
	}
	return ret;
}

void random_walk_search(double goal, long n){
	double *k = new double[n];
	double *s = new double[n];
	double *ktmp = new double[n];
	double *stmp = new double[n];
	for (long i = 0; i < n; i++){
		k[i] = 1.0;
		s[i] = 0.5;
	}
	k[n-1] = 10.0;
	split_res best = split(k, s, n);
	uint64_t count = -1;
	while (true){
		count++;
		for (long i = 0; i < 1000000; i++){
			Vec<double> err = random_vec(n*2, 0.0001/(count+1));
			for (long i = 0; i < n; i++){
				ktmp[i] = k[i] * (1 + err[i]);
				stmp[i] = s[i] + err[i+n];
			}
			split_res res = split(ktmp, stmp, n);
			if ((res.sol/res.fal > goal) && (res.sol / res.cost > best.sol / best.cost)){
				for (long i = 0; i < n; i++){
					k[i] = k[i] * (1 + err[i]);
					s[i] = s[i] + err[i+n];
				}
				best.cost = res.cost; 
				best.sol = res.sol; 
				best.fal = res.fal;
			}
		}
		printf("[epoch %ld] ", count);
		cout << "k = [";
		for (long i = 0; i < n-1; i++){
			printf("%.5f ", k[i]);
		}
		printf("%.5f]", k[n-1]);
		cout << ", s = [";
		for (long i = 0; i < n-1; i++){
			printf("%.5f ", s[i]);
		}
		printf("%.5f]", s[n-1]);
		printf(", cost = %.5f, sol = %.5f, fal = %.5f\n", best.cost, best.sol, best.fal);
	}
}



int main(int argc, char *argv[]){
	random_walk_search(atof(argv[2]), atol(argv[1]));
	return 0;
}
