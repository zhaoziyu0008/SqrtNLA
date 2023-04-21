#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <NTL/LLL.h>
#include <lattice.h>
#include <pool.h>



using namespace std;
using namespace NTL;

#define DTG_BY_L2NORM true
#define NUM_DTG_STEP 3
#define FIRST_STAGE_BLOCKSIZE 32
#define SECOND_STAGE_BLOCKSIZE 128
#define SECOND_STAGE_VECLENGTH 1280
#define FINAL_STAGE_BLOCKSIZE 512
#define FINAL_STAGE_VECLENGTH 640

struct PJT_DTG_strategy{
    long num_step;
    double *k;
    double *s;
};

struct PJT_DTG_sol_bucket{
    long *solp;
    long *soln;
    long num_sol_p;
    long num_sol_n;
};

double get_mean_gap(double dualtest_alpha){
    return 0.001124;
}
double get_one_sigma_num(double mean_gap){
    return 1. / (180. * mean_gap * mean_gap);
}

float add_round_norm(float *src1, float *src2, long n){
    __m512 absp1 = _mm512_setzero_ps();
    __m512 absp2 = _mm512_setzero_ps();
    __m512 absp3 = _mm512_setzero_ps();
    __m512 absp4 = _mm512_setzero_ps();
    for (long i = 0; i < n; i+=64){
        __m512 x1 = _mm512_load_ps(src1+i);
        __m512 x2 = _mm512_load_ps(src1+i+16);
        __m512 x3 = _mm512_load_ps(src1+i+32);
        __m512 x4 = _mm512_load_ps(src1+i+48);
        __m512 y1 = _mm512_load_ps(src2+i);
        __m512 y2 = _mm512_load_ps(src2+i+16);
        __m512 y3 = _mm512_load_ps(src2+i+32);
        __m512 y4 = _mm512_load_ps(src2+i+48);
        _mm_prefetch((const char*)(&src1[i+256]), _MM_HINT_T1);
        _mm_prefetch((const char*)(&src2[i+256]), _MM_HINT_T1);

        __m512 sum1 = _mm512_add_ps(x1, y1);
        __m512 sum2 = _mm512_add_ps(x2, y2);
        __m512 sum3 = _mm512_add_ps(x3, y3);
        __m512 sum4 = _mm512_add_ps(x4, y4);

        sum1 = _mm512_reduce_ps(sum1, 0);
        sum2 = _mm512_reduce_ps(sum2, 0);
        sum3 = _mm512_reduce_ps(sum3, 0);
        sum4 = _mm512_reduce_ps(sum4, 0);  

        absp1 = _mm512_fmadd_ps(sum1, sum1, absp1);
        absp2 = _mm512_fmadd_ps(sum2, sum2, absp2);
        absp3 = _mm512_fmadd_ps(sum3, sum3, absp3);
        absp4 = _mm512_fmadd_ps(sum4, sum4, absp4);
    }
    absp1 = _mm512_add_ps(absp1, absp2);
    absp3 = _mm512_add_ps(absp3, absp4);
    absp1 = _mm512_add_ps(absp1, absp3);
    return _mm512_reduce_add_ps(absp1);
}

float sub_round_norm(float *src1, float *src2, long n){
    __m512 absn1 = _mm512_setzero_ps();
    __m512 absn2 = _mm512_setzero_ps();
    __m512 absn3 = _mm512_setzero_ps();
    __m512 absn4 = _mm512_setzero_ps();
    for (long i = 0; i < n; i+=64){
        __m512 x1 = _mm512_load_ps(src1+i);
        __m512 x2 = _mm512_load_ps(src1+i+16);
        __m512 x3 = _mm512_load_ps(src1+i+32);
        __m512 x4 = _mm512_load_ps(src1+i+48);
        __m512 y1 = _mm512_load_ps(src2+i);
        __m512 y2 = _mm512_load_ps(src2+i+16);
        __m512 y3 = _mm512_load_ps(src2+i+32);
        __m512 y4 = _mm512_load_ps(src2+i+48);
        _mm_prefetch((const char*)(&src1[i+256]), _MM_HINT_T1);
        _mm_prefetch((const char*)(&src2[i+256]), _MM_HINT_T1);

        __m512 sub1 = _mm512_sub_ps(x1, y1);
        __m512 sub2 = _mm512_sub_ps(x2, y2);
        __m512 sub3 = _mm512_sub_ps(x3, y3);
        __m512 sub4 = _mm512_sub_ps(x4, y4);
 
        sub1 = _mm512_reduce_ps(sub1, 0);
        sub2 = _mm512_reduce_ps(sub2, 0);
        sub3 = _mm512_reduce_ps(sub3, 0);
        sub4 = _mm512_reduce_ps(sub4, 0);

        absn1 = _mm512_fmadd_ps(sub1, sub1, absn1);
        absn2 = _mm512_fmadd_ps(sub2, sub2, absn2);
        absn3 = _mm512_fmadd_ps(sub3, sub3, absn3);
        absn4 = _mm512_fmadd_ps(sub4, sub4, absn4);
    }
    absn1 = _mm512_add_ps(absn1, absn2);
    absn3 = _mm512_add_ps(absn3, absn4);
    absn1 = _mm512_add_ps(absn1, absn3);
    return _mm512_reduce_add_ps(absn1);
}




//try to find a short vec.
int main (int argc, char *argv[]){
    Lattice_QP L(argv[1]);
    long num_threads = 8;
    bool show_details = true;
    cpu_set_t *mask = new cpu_set_t[num_threads];
    for (long i = 0; i < num_threads; i++){
        CPU_ZERO(&mask[i]);
        CPU_SET(i, &mask[i]);
    }
    struct timeval run_start, run_end;
	gettimeofday(&run_start, NULL);



    /* analyze the basis and fix the params, currently 
     * all params are set by hand, we will write an 
     * auto params optimizing funtion instead of it. */
    if (show_details) printf("params analyzing begin...\n");
    long d = 70;                        //dual dimension
    double goal_length = 880000.0;      //goal norm
    double expect_tail_length = 394000.0;
    double expect_dual_length = 4.95e-6;
    double expect_improve_ratio = 1014630.0/sqrt(goal_length*goal_length-expect_tail_length*expect_tail_length);
    double expect_dualtest_alpha = expect_dual_length * goal_length / sqrt(d);
    double expect_mean_gap = get_mean_gap(expect_dualtest_alpha);
    double expect_one_sigma_num = get_one_sigma_num(expect_mean_gap);
    PJT_DTG_strategy strategy;
    strategy.num_step = NUM_DTG_STEP;
    strategy.k = new double[strategy.num_step];
    strategy.s = new double[strategy.num_step];
    strategy.k[0] = 1.38904; strategy.k[1] = 3.23934; strategy.k[2] = 6.39166;
    strategy.s[0] = -0.02257; strategy.s[1] = 1.21797; strategy.s[2] = 2.38714;    
    
    long num_st = 11264;            //we only use 2 combs now, need to be divided by 32
    long msd_st = 80;               //msd to get short tail vectors
    long dim_st = 80;               //full svp dim to get short tail vectors

    long num_dt[3] = {8448, 46080, 179200};                 //vectors to do dual test
    double dt_bound[3] = {0.0821911, 0.082632+0.0001, 0.082629+0.0002};    //avg norm to pass dual test
    if (show_details){
        printf("params analyzing done. d = %ld, goal_length = %ld\n", d, (long)goal_length);
        printf("expect_tail_length = %ld, expect_dual_length = 1/%ld, " 
               "expect improve ratio = %.4f, expect_dualtest_alpha = %.4f, "
               "expect_mean_gap = %.6f, expect_one_sigma_sum = %ld\n",
               (long)expect_tail_length, (long)(1./expect_dual_length),
               expect_improve_ratio, expect_dualtest_alpha, expect_mean_gap, (long)expect_one_sigma_num);
        printf("distinguish strategy: num_step = %ld, k = [%.4f, %.4f, %.4f], s = [%.4f, %.4f, %.4f]\n\n", 
                strategy.num_step, strategy.k[0], strategy.k[1], strategy.k[2], strategy.s[0], strategy.s[1], strategy.s[2]);
    }



    /* find short vec in the dual part, and store it in 
     * dt_vecs and dc_vecs */
    if (show_details) printf("sieving for short dual vecs begin...\n");
    struct timeval dts_start, dts_end;
	gettimeofday(&dts_start, NULL);
    Lattice_QP Ld = L.b_loc_QP(0, d);
    Lattice_QP Ld_dual = Ld.dual_QP();
    Ld_dual.LLL_QP(0.99);
    Ld_dual.LLL_DEEP_QP(0.99);
    Pool dp(&Ld_dual);
    dp.left_progressive_sieve_num(num_dt[2]+num_dt[1]+num_dt[0], 0, d, num_threads, 2-2);
    float ***dt_vecs = new float**[3];
    dt_vecs[0] = dp.get_short_vecs_fp(num_dt[0]);
    dt_vecs[1] = dp.get_short_vecs_fp(num_dt[1]+num_dt[0]);
    dt_vecs[2] = dp.get_short_vecs_fp(num_dt[2]+num_dt[1]+num_dt[0]);
    dp.clear_all();
    gettimeofday(&dts_end, NULL);
    double dts_time = (dts_end.tv_sec-dts_start.tv_sec)+(double)(dts_end.tv_usec-dts_start.tv_usec)/1000000.0;
    if (show_details){
        /*  cout << "[";
            for (long i = 0; i < num_dt; i++){
                double x = dot(dt_vecs[i], dt_vecs[i], d);
                cout << sqrt(x) << " ";
            }
            cout << "]\n";
        */
        long _avglength0 = (long)sqrt(1.0/dot(dt_vecs[0][num_dt[0]/2], dt_vecs[0][num_dt[0]/2], d));
        long _avglength1 = (long)sqrt(1.0/dot(dt_vecs[1][num_dt[1]/2], dt_vecs[1][num_dt[1]/2], d));
        long _avglength2 = (long)sqrt(1.0/dot(dt_vecs[2][num_dt[2]/2], dt_vecs[2][num_dt[2]/2], d));
        long _minlength0 = (long)sqrt(1.0/dot(dt_vecs[0][0], dt_vecs[0][0], d));
        long _minlength1 = (long)sqrt(1.0/dot(dt_vecs[1][0], dt_vecs[1][0], d));
        long _minlength2 = (long)sqrt(1.0/dot(dt_vecs[2][0], dt_vecs[2][0], d));
        long _maxlength0 = (long)sqrt(1.0/dot(dt_vecs[0][num_dt[0]-1], dt_vecs[0][num_dt[0]-1], d));
        long _maxlength1 = (long)sqrt(1.0/dot(dt_vecs[1][num_dt[1]-1], dt_vecs[1][num_dt[1]-1], d));
        long _maxlength2 = (long)sqrt(1.0/dot(dt_vecs[2][num_dt[2]-1], dt_vecs[2][num_dt[2]-1], d));
        printf("done. time = %fs, 1/avg_length = [%ld, %ld, %ld], 1/min_length = [%ld, %ld, %ld], 1/max_length = [%ld, %ld, %ld]\n\n"
                , dts_time, _avglength0, _avglength1, _avglength2, _minlength0, _minlength1, _minlength2, _maxlength0, _maxlength1, _maxlength2);
    }
    


    /* find some vecs with short tail norm, and store it 
     * in st_vecs */
    struct timeval sts_start, sts_end;
	gettimeofday(&sts_start, NULL);
    if (show_details) printf("sieving for short tail vecs begin...\n");
    Lattice_QP Lw = L.b_loc_QP(0, d + dim_st);
    Lattice_QP Lw_dual = Lw.dual_QP();
    Pool wp(&Lw);
    wp.left_progressive_sieve(d+dim_st-msd_st, d+dim_st, num_threads, 2-2);
    while(wp.index_l > d) wp.extend_left();
    wp.sort_cvec();
    double **st_vecs = wp.get_short_vecs(num_st);
    do {
        double **bw = Lw.get_b().hi;
        double *Bw = Lw.get_B().hi;
        double *Bwsi = (double *) NEW_VEC(d, sizeof(double));
        for (long i = 0; i < d; i++) Bwsi[i] = 1.0/sqrt(Bw[i]); 
        #pragma omp parallel for
        for (long i = 0; i < num_st; i++){
            for (long j = d-1; j >= 0; j--){
                long q = round(st_vecs[i][j]*Bwsi[j]);
                red(st_vecs[i], bw[j], q, j+1);
            }
        }
    }while(0);
    wp.clear_all();
    float **st_vecs_fp = f64_to_f32(st_vecs, num_st, d);
    gettimeofday(&sts_end, NULL);
    double sts_time = (sts_end.tv_sec-sts_start.tv_sec)+(double)(sts_end.tv_usec-sts_start.tv_usec)/1000000.0;
    if (show_details){
        /*  for (long i = 0; i < num_st; i++){
                double x = 0.0;
                for (long j = 0; j < dim_st; j++){
                    x += st_vecs[i][d+j]*st_vecs[i][d+j];
                }
                cout << sqrt(2*x) << " ";
            }
            cout << "]\n";
        */
        double min_norm = 1e100;
        double max_norm = 0.0;
        double avg_norm = 0.0;
        for (long i = 0; i < num_st; i++){
            double norm = 0.0;
            for (long j = d; j < d+dim_st; j++){
                norm += st_vecs[i][j]*st_vecs[i][j];
            }
            if (norm > max_norm) max_norm = norm;
            if (norm < min_norm) min_norm = norm;
            avg_norm += norm;
        }
        avg_norm /= num_st;
        printf("done. time = %fs, avg_length = %ld, min_length = %ld, max_length = %ld\n\n", 
                sts_time, (long)sqrt(avg_norm), (long)sqrt(min_norm), (long)sqrt(max_norm));
    }
    
    /* find short vecs that are 2 or 3 combs 
     * of the short vecs find above, a full 
     * search or some other techniques? we 
     * now only naively search 2 combs */
    // THE FIRST STAGE
    if (show_details) {
        printf("the first stage of filtering begin, %ld pairs will be test by %ld dual vecs, bound = %.5f, %ld pairs expect to pass\n",
                num_st*(num_st-1), num_dt[0], dt_bound[0], (long)((1-erf((strategy.k[0]-strategy.s[0])/sqrt(2.)))*0.5*num_st*(num_st-1)));
        printf("begin create dot_mat...");
        std::cout << std::flush;
    }
    struct timeval dt0_start, dt0_end;
	gettimeofday(&dt0_start, NULL);
    float **dot_mat0 = (float **) NEW_MAT(num_st, num_dt[0], sizeof(float));
    #pragma omp parallel for 
    for (long i = 0; i < num_st; i++){
        for (long j = 0; j < num_dt[0]; j++){
            dot_mat0[i][j] = dot(st_vecs_fp[i], dt_vecs[0][j], d);
        }
    }
    gettimeofday(&dt0_end, NULL);
    double dt0_dm_time = (dt0_end.tv_sec-dt0_start.tv_sec)+(double)(dt0_end.tv_usec-dt0_start.tv_usec)/1000000.0;
    if (show_details) printf("done, time = %fs\n", dt0_dm_time);
    gettimeofday(&dt0_start, NULL);
    if (show_details) printf("begin first round dual test...\n");
    PJT_DTG_sol_bucket **sol_table0;
    PJT_DTG_sol_bucket *buf0;
    float dt0_bound = dt_bound[0] * num_dt[0];
    do {
        //malloc
        sol_table0 = new PJT_DTG_sol_bucket*[num_st/SECOND_STAGE_BLOCKSIZE];
        for (long i = 0; i < num_st/SECOND_STAGE_BLOCKSIZE; i++){
            sol_table0[i] = new PJT_DTG_sol_bucket[num_st/SECOND_STAGE_BLOCKSIZE-i];
        }
        buf0 = new PJT_DTG_sol_bucket[num_threads];
        for (long i = 0; i < num_threads; i++){
            buf0[i].solp = (long *) NEW_VEC(SECOND_STAGE_BLOCKSIZE * SECOND_STAGE_BLOCKSIZE, sizeof(long));
            buf0[i].soln = (long *) NEW_VEC(SECOND_STAGE_BLOCKSIZE * SECOND_STAGE_BLOCKSIZE, sizeof(long));
        }
    }while(0);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (long ind = 0; ind < num_st/SECOND_STAGE_BLOCKSIZE; ind++){
        for (long jnd = 0; jnd < num_st/SECOND_STAGE_BLOCKSIZE; jnd++){
            if (jnd < ind) continue;
            pthread_setaffinity_np(pthread_self(), sizeof(mask[omp_get_thread_num()]), &mask[omp_get_thread_num()]);
            PJT_DTG_sol_bucket *buffer = &buf0[omp_get_thread_num()];
            buffer->num_sol_p = 0;
            buffer->num_sol_n = 0;
            for (long ibegin = ind * SECOND_STAGE_BLOCKSIZE; ibegin < ind * SECOND_STAGE_BLOCKSIZE + SECOND_STAGE_BLOCKSIZE; ibegin+=FIRST_STAGE_BLOCKSIZE){
                for (long jbegin = jnd * SECOND_STAGE_BLOCKSIZE; jbegin < jnd * SECOND_STAGE_BLOCKSIZE + SECOND_STAGE_BLOCKSIZE; jbegin += FIRST_STAGE_BLOCKSIZE){
                    if (jbegin > ibegin){
                        for (long i = ibegin; i < ibegin+FIRST_STAGE_BLOCKSIZE; i++){
                            for (long j = jbegin; j < jbegin+FIRST_STAGE_BLOCKSIZE; j++){
                                float *src1 = dot_mat0[i];
                                float *src2 = dot_mat0[j];
                                __m512 absp1 = _mm512_setzero_ps();
                                __m512 absp2 = _mm512_setzero_ps();
                                __m512 absp3 = _mm512_setzero_ps();
                                __m512 absp4 = _mm512_setzero_ps();
                                __m512 absn1 = _mm512_setzero_ps();
                                __m512 absn2 = _mm512_setzero_ps();
                                __m512 absn3 = _mm512_setzero_ps();
                                __m512 absn4 = _mm512_setzero_ps();
                                for (long i = 0; i < num_dt[0]; i+=64){
                                    __m512 x1 = _mm512_load_ps(src1+i);
                                    __m512 x2 = _mm512_load_ps(src1+i+16);
                                    __m512 x3 = _mm512_load_ps(src1+i+32);
                                    __m512 x4 = _mm512_load_ps(src1+i+48);
                                    __m512 y1 = _mm512_load_ps(src2+i);
                                    __m512 y2 = _mm512_load_ps(src2+i+16);
                                    __m512 y3 = _mm512_load_ps(src2+i+32);
                                    __m512 y4 = _mm512_load_ps(src2+i+48);

                                    __m512 sum1 = _mm512_add_ps(x1, y1);
                                    __m512 sum2 = _mm512_add_ps(x2, y2);
                                    __m512 sum3 = _mm512_add_ps(x3, y3);
                                    __m512 sum4 = _mm512_add_ps(x4, y4);
                                    __m512 sub1 = _mm512_sub_ps(x1, y1);
                                    __m512 sub2 = _mm512_sub_ps(x2, y2);
                                    __m512 sub3 = _mm512_sub_ps(x3, y3);
                                    __m512 sub4 = _mm512_sub_ps(x4, y4);

                                    sum1 = _mm512_reduce_ps(sum1, 0);
                                    sum2 = _mm512_reduce_ps(sum2, 0);
                                    sum3 = _mm512_reduce_ps(sum3, 0);
                                    sum4 = _mm512_reduce_ps(sum4, 0);  
                                    sub1 = _mm512_reduce_ps(sub1, 0);
                                    sub2 = _mm512_reduce_ps(sub2, 0);
                                    sub3 = _mm512_reduce_ps(sub3, 0);
                                    sub4 = _mm512_reduce_ps(sub4, 0);

                                    absp1 = _mm512_fmadd_ps(sum1, sum1, absp1);
                                    absp2 = _mm512_fmadd_ps(sum2, sum2, absp2);
                                    absp3 = _mm512_fmadd_ps(sum3, sum3, absp3);
                                    absp4 = _mm512_fmadd_ps(sum4, sum4, absp4);
                                    absn1 = _mm512_fmadd_ps(sub1, sub1, absn1);
                                    absn2 = _mm512_fmadd_ps(sub2, sub2, absn2);
                                    absn3 = _mm512_fmadd_ps(sub3, sub3, absn3);
                                    absn4 = _mm512_fmadd_ps(sub4, sub4, absn4);
                                }
                                absp1 = _mm512_add_ps(absp1, absp2);
                                absp3 = _mm512_add_ps(absp3, absp4);
                                absn1 = _mm512_add_ps(absn1, absn2);
                                absn3 = _mm512_add_ps(absn3, absn4);
                                absp1 = _mm512_add_ps(absp1, absp3);
                                absn1 = _mm512_add_ps(absn1, absn3);
                                float resp = _mm512_reduce_add_ps(absp1);
                                float resn = _mm512_reduce_add_ps(absn1);
                                if (resp < dt0_bound){
                                    buffer->solp[buffer->num_sol_p] = i;
                                    buffer->solp[buffer->num_sol_p+1] = j;
                                    buffer->num_sol_p+=2;
                                }
                                if (resn < dt0_bound){
                                    buffer->soln[buffer->num_sol_n] = i;
                                    buffer->soln[buffer->num_sol_n+1] = j;
                                    buffer->num_sol_n+=2;
                                }
                            }
                        }
                    }
                    if (jbegin == ibegin){
                        for (long i = ibegin; i < ibegin+FIRST_STAGE_BLOCKSIZE; i++){
                            for (long j = i+1; j < jbegin+FIRST_STAGE_BLOCKSIZE; j++){
                                float *src1 = dot_mat0[i];
                                float *src2 = dot_mat0[j];
                                __m512 absp1 = _mm512_setzero_ps();
                                __m512 absp2 = _mm512_setzero_ps();
                                __m512 absp3 = _mm512_setzero_ps();
                                __m512 absp4 = _mm512_setzero_ps();
                                __m512 absn1 = _mm512_setzero_ps();
                                __m512 absn2 = _mm512_setzero_ps();
                                __m512 absn3 = _mm512_setzero_ps();
                                __m512 absn4 = _mm512_setzero_ps();
                                for (long i = 0; i < num_dt[0]; i+=64){
                                    __m512 x1 = _mm512_load_ps(src1+i);
                                    __m512 x2 = _mm512_load_ps(src1+i+16);
                                    __m512 x3 = _mm512_load_ps(src1+i+32);
                                    __m512 x4 = _mm512_load_ps(src1+i+48);
                                    __m512 y1 = _mm512_load_ps(src2+i);
                                    __m512 y2 = _mm512_load_ps(src2+i+16);
                                    __m512 y3 = _mm512_load_ps(src2+i+32);
                                    __m512 y4 = _mm512_load_ps(src2+i+48);

                                    __m512 sum1 = _mm512_add_ps(x1, y1);
                                    __m512 sum2 = _mm512_add_ps(x2, y2);
                                    __m512 sum3 = _mm512_add_ps(x3, y3);
                                    __m512 sum4 = _mm512_add_ps(x4, y4);
                                    __m512 sub1 = _mm512_sub_ps(x1, y1);
                                    __m512 sub2 = _mm512_sub_ps(x2, y2);
                                    __m512 sub3 = _mm512_sub_ps(x3, y3);
                                    __m512 sub4 = _mm512_sub_ps(x4, y4);

                                    sum1 = _mm512_reduce_ps(sum1, 0);
                                    sum2 = _mm512_reduce_ps(sum2, 0);
                                    sum3 = _mm512_reduce_ps(sum3, 0);
                                    sum4 = _mm512_reduce_ps(sum4, 0);  
                                    sub1 = _mm512_reduce_ps(sub1, 0);
                                    sub2 = _mm512_reduce_ps(sub2, 0);
                                    sub3 = _mm512_reduce_ps(sub3, 0);
                                    sub4 = _mm512_reduce_ps(sub4, 0);

                                    absp1 = _mm512_fmadd_ps(sum1, sum1, absp1);
                                    absp2 = _mm512_fmadd_ps(sum2, sum2, absp2);
                                    absp3 = _mm512_fmadd_ps(sum3, sum3, absp3);
                                    absp4 = _mm512_fmadd_ps(sum4, sum4, absp4);
                                    absn1 = _mm512_fmadd_ps(sub1, sub1, absn1);
                                    absn2 = _mm512_fmadd_ps(sub2, sub2, absn2);
                                    absn3 = _mm512_fmadd_ps(sub3, sub3, absn3);
                                    absn4 = _mm512_fmadd_ps(sub4, sub4, absn4);
                                }
                                absp1 = _mm512_add_ps(absp1, absp2);
                                absp3 = _mm512_add_ps(absp3, absp4);
                                absn1 = _mm512_add_ps(absn1, absn2);
                                absn3 = _mm512_add_ps(absn3, absn4);
                                absp1 = _mm512_add_ps(absp1, absp3);
                                absn1 = _mm512_add_ps(absn1, absn3);
                                float resp = _mm512_reduce_add_ps(absp1);
                                float resn = _mm512_reduce_add_ps(absn1);
                                if (resp < dt0_bound){
                                    buffer->solp[buffer->num_sol_p] = i;
                                    buffer->solp[buffer->num_sol_p+1] = j;
                                    buffer->num_sol_p+=2;
                                }
                                if (resn < dt0_bound){
                                    buffer->soln[buffer->num_sol_n] = i;
                                    buffer->soln[buffer->num_sol_n+1] = j;
                                    buffer->num_sol_n+=2;
                                }
                            }
                        }
                    }
                }
            }
            sol_table0[ind][jnd-ind].num_sol_p = buffer->num_sol_p;
            sol_table0[ind][jnd-ind].num_sol_n = buffer->num_sol_n;
            sol_table0[ind][jnd-ind].solp = new long[buffer->num_sol_p];
            sol_table0[ind][jnd-ind].soln = new long[buffer->num_sol_n];
            for (long i = 0; i < buffer->num_sol_p; i++) sol_table0[ind][jnd-ind].solp[i] = buffer->solp[i];
            for (long i = 0; i < buffer->num_sol_n; i++) sol_table0[ind][jnd-ind].soln[i] = buffer->soln[i];
        }
    }
    do {
        //free
        for (long i = 0; i < num_threads; i++){
            FREE_VEC(buf0[i].solp);
            FREE_VEC(buf0[i].soln);
        }
        delete[] buf0;
    } while(0);
    gettimeofday(&dt0_end, NULL);
    double dt0_time = (dt0_end.tv_sec-dt0_start.tv_sec)+(double)(dt0_end.tv_usec-dt0_start.tv_usec)/1000000.0;
    long num_sol_stage0;
    if (show_details){
        long num_flops = num_st*(num_st-1)*num_dt[0];
        num_sol_stage0 = 0;
        for (long i = 0; i < num_st/SECOND_STAGE_BLOCKSIZE; i++){
            for (long j = i; j < num_st/SECOND_STAGE_BLOCKSIZE; j++){
                num_sol_stage0 += sol_table0[i][j-i].num_sol_p;
                num_sol_stage0 += sol_table0[i][j-i].num_sol_n;
            }
        }
        num_sol_stage0/=2;
        printf("done. time = %fs, found %ld pairs in total, speed = %f MFLOPS\n\n", dt0_time, num_sol_stage0, num_flops/(dt0_time*1000000));
    }


    // THE SECOND STAGE
    if (show_details) {
        printf("the second stage of filtering begin, %ld pairs will be test by %ld dual vecs, bound = %.5f, %ld pairs expect to pass\n",
                num_sol_stage0, num_dt[1], dt_bound[1], (long)((1-erf((strategy.k[1]-strategy.s[1])/sqrt(2.)))*0.5*num_sol_stage0));
        printf("begin create dot_mat...");
        std::cout << std::flush;
    }
    struct timeval dt1_start, dt1_end;
	gettimeofday(&dt1_start, NULL);
    float **dot_mat1 = (float **) NEW_MAT(num_st, num_dt[1], sizeof(float));
    #pragma omp parallel for 
    for (long i = 0; i < num_st; i++){
        for (long j = num_dt[0]; j < num_dt[1]+num_dt[0]; j++){
            dot_mat1[i][j-num_dt[0]] = dot(st_vecs_fp[i], dt_vecs[1][j], d);
        }
        //copy(dot_mat1[i], dot_mat0[i], num_dt[0]);
        //for (long j = num_dt[0]; j < num_dt[1]; j++){
        //    dot_mat1[i][j] = dot(st_vecs_fp[i], dt_vecs[1][j], d);
        //}
    }
    gettimeofday(&dt1_end, NULL);
    double dt1_dm_time = (dt1_end.tv_sec-dt1_start.tv_sec)+(double)(dt1_end.tv_usec-dt1_start.tv_usec)/1000000.0;
    if (show_details) printf("done, time = %fs\n", dt1_dm_time);

    gettimeofday(&dt1_start, NULL);
    if (show_details) printf("begin second round dual test...\n");
    PJT_DTG_sol_bucket **sol_table1;
    PJT_DTG_sol_bucket *buf1;
    float ***fp1_buf;
    float dt1_bound = dt_bound[1] * num_dt[1];
    do {
        //malloc
        sol_table1 = new PJT_DTG_sol_bucket*[num_st/SECOND_STAGE_BLOCKSIZE];
        for (long i = 0; i < num_st/SECOND_STAGE_BLOCKSIZE; i++){
            sol_table1[i] = new PJT_DTG_sol_bucket[num_st/SECOND_STAGE_BLOCKSIZE-i];
        }
        buf1 = new PJT_DTG_sol_bucket[num_threads];
        for (long i = 0; i < num_threads; i++){
            buf1[i].solp = (long *) NEW_VEC(SECOND_STAGE_BLOCKSIZE*SECOND_STAGE_BLOCKSIZE*2, sizeof(long));
            buf1[i].soln = (long *) NEW_VEC(SECOND_STAGE_BLOCKSIZE*SECOND_STAGE_BLOCKSIZE*2, sizeof(long));
        }
        fp1_buf = new float**[num_threads];
        for (long i = 0; i < num_threads; i++) fp1_buf[i] = (float **) NEW_MAT(2, SECOND_STAGE_BLOCKSIZE * SECOND_STAGE_BLOCKSIZE, sizeof(float));
    }while(0);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (long ind = 0; ind < num_st/SECOND_STAGE_BLOCKSIZE; ind++){
        for (long jnd = 0; jnd < num_st/SECOND_STAGE_BLOCKSIZE; jnd++){
            if (jnd < ind) continue;
            pthread_setaffinity_np(pthread_self(), sizeof(mask[omp_get_thread_num()]), &mask[omp_get_thread_num()]);
            PJT_DTG_sol_bucket *buffer = &buf1[omp_get_thread_num()];
            PJT_DTG_sol_bucket *to_check = &sol_table0[ind][jnd-ind];
            float **fp_buf = fp1_buf[omp_get_thread_num()];
            set_zero(fp_buf[0], to_check->num_sol_p);
            set_zero(fp_buf[1], to_check->num_sol_n);
            buffer->num_sol_p = 0;
            buffer->num_sol_n = 0;
            for (long bias = 0; bias < num_dt[1]; bias += SECOND_STAGE_VECLENGTH){
                for (long l = 0; l < to_check->num_sol_p/2; l++){
                    float *src1 = dot_mat1[to_check->solp[l*2]]+bias;
                    float *src2 = dot_mat1[to_check->solp[l*2+1]]+bias;
                    fp_buf[0][l] += add_round_norm(src1, src2, SECOND_STAGE_VECLENGTH);
                }
                for (long l = 0; l < to_check->num_sol_n/2; l++){
                    float *src1 = dot_mat1[to_check->soln[l*2]]+bias;
                    float *src2 = dot_mat1[to_check->soln[l*2+1]]+bias;
                    fp_buf[1][l] += sub_round_norm(src1, src2, SECOND_STAGE_VECLENGTH);
                }
            }
            for (long l = 0; l < to_check->num_sol_p/2; l++){
                if (fp_buf[0][l] < dt1_bound){
                    buffer->solp[buffer->num_sol_p] = to_check->solp[l*2];
                    buffer->solp[buffer->num_sol_p+1] = to_check->solp[l*2+1];
                    buffer->num_sol_p+=2;
                }
            }
            for (long l = 0; l < to_check->num_sol_n/2; l++){
                if (fp_buf[1][l] < dt1_bound){
                    buffer->soln[buffer->num_sol_n] = to_check->soln[l*2];
                    buffer->soln[buffer->num_sol_n+1] = to_check->soln[l*2+1];
                    buffer->num_sol_n+=2;
                }
            }

            sol_table1[ind][jnd-ind].num_sol_p = buffer->num_sol_p;
            sol_table1[ind][jnd-ind].num_sol_n = buffer->num_sol_n;
            sol_table1[ind][jnd-ind].solp = new long[buffer->num_sol_p];
            sol_table1[ind][jnd-ind].soln = new long[buffer->num_sol_n];
            for (long i = 0; i < buffer->num_sol_p; i++) sol_table1[ind][jnd-ind].solp[i] = buffer->solp[i];
            for (long i = 0; i < buffer->num_sol_n; i++) sol_table1[ind][jnd-ind].soln[i] = buffer->soln[i];
        }
    }
    do {
        //free
        for (long i = 0; i < num_st/SECOND_STAGE_BLOCKSIZE; i++){
            for (long j = 0; j < num_st/SECOND_STAGE_BLOCKSIZE - i; j++){
                delete[] sol_table0[i][j].solp;
                delete[] sol_table0[i][j].soln;
            }
            delete[] sol_table0[i];
        }
        delete[] sol_table0;
        for (long i = 0; i < num_threads; i++){
            FREE_VEC(buf1[i].solp);
            FREE_VEC(buf1[i].soln);
            FREE_MAT(fp1_buf[i]);
        }
        FREE_MAT(dot_mat0);
        delete[] fp1_buf;
        delete[] buf1;
    } while(0);

    gettimeofday(&dt1_end, NULL);
    double dt1_time = (dt1_end.tv_sec-dt1_start.tv_sec)+(double)(dt1_end.tv_usec-dt1_start.tv_usec)/1000000.0;
    long num_sol_stage1;
    if (show_details){
        long num_flops = num_sol_stage0*num_dt[1];
        num_sol_stage1 = 0;
        for (long i = 0; i < num_st/SECOND_STAGE_BLOCKSIZE; i++){
            for (long j = i; j < num_st/SECOND_STAGE_BLOCKSIZE; j++){
                num_sol_stage1 += sol_table1[i][j-i].num_sol_p;
                num_sol_stage1 += sol_table1[i][j-i].num_sol_n;
            }
        }
        num_sol_stage1 /= 2;
        //printf("done. time = %fs, found %ld pairs in total, speed = %f MFLOPS\n\n", dt1_time, num_sol_stage1, num_flops/(dt1_time*1000000));
    }

    // THE FINAL STAGE
    if (show_details) {
        printf("final stage of filtering begin, %ld pairs will be test by %ld dual vecs, bound = %.5f, %ld pairs expect to pass\n",
                num_sol_stage1, num_dt[2], dt_bound[2], (long)((1-erf((strategy.k[2]-strategy.s[2])/sqrt(2.)))*0.5*num_sol_stage1));
        printf("begin create dot_mat...");
        std::cout << std::flush;
    }
    struct timeval dt2_start, dt2_end;
	gettimeofday(&dt2_start, NULL);
    float **dot_mat2 = (float **) NEW_MAT(num_st, num_dt[2], sizeof(float));
    #pragma omp parallel for 
    for (long i = 0; i < num_st; i++){
        for (long j = num_dt[1]+num_dt[0]; j < num_dt[2]+num_dt[1]+num_dt[0]; j++){
            dot_mat2[i][j-num_dt[1]-num_dt[0]] = dot(st_vecs_fp[i], dt_vecs[2][j], d);
        }
        //copy(dot_mat2[i], dot_mat1[i], num_dt[1]);
        //for (long j = num_dt[1]; j < num_dt[2]; j++){
        //    dot_mat2[i][j] = dot(st_vecs_fp[i], dt_vecs[2][j], d);
        //}
    }
    gettimeofday(&dt2_end, NULL);
    double dt2_dm_time = (dt2_end.tv_sec-dt2_start.tv_sec)+(double)(dt2_end.tv_usec-dt2_start.tv_usec)/1000000.0;
    if (show_details) printf("done, time = %fs\n", dt2_dm_time);
    gettimeofday(&dt2_start, NULL);
    if (show_details) printf("begin final round dual test...\n");
    PJT_DTG_sol_bucket *sol_list;
    float ***fp2_buf;
    float dt2_bound = dt_bound[2] * num_dt[2];
    do {
        //malloc
        sol_list = new PJT_DTG_sol_bucket[num_threads];
        for (long i = 0; i < num_threads; i++){
            sol_list[i].num_sol_p = 0;
            sol_list[i].num_sol_n = 0;
            sol_list[i].solp = (long *) NEW_VEC(SECOND_STAGE_BLOCKSIZE*SECOND_STAGE_BLOCKSIZE*2, sizeof(long));
            sol_list[i].soln = (long *) NEW_VEC(SECOND_STAGE_BLOCKSIZE*SECOND_STAGE_BLOCKSIZE*2, sizeof(long));
        }
        fp2_buf = new float**[num_threads];
        for (long i = 0; i < num_threads; i++) fp1_buf[i] = (float **) NEW_MAT(2, 16 * SECOND_STAGE_BLOCKSIZE * SECOND_STAGE_BLOCKSIZE, sizeof(float));
    }while(0);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (long ind = 0; ind < num_st/FINAL_STAGE_BLOCKSIZE; ind++){
        for (long jnd = 0; jnd < num_st/FINAL_STAGE_BLOCKSIZE; jnd++){
            if (jnd < ind) continue;
            pthread_setaffinity_np(pthread_self(), sizeof(mask[omp_get_thread_num()]), &mask[omp_get_thread_num()]);
            PJT_DTG_sol_bucket *buffer = &sol_list[omp_get_thread_num()];
            float **fp_buf = fp2_buf[omp_get_thread_num()];
            PJT_DTG_sol_bucket to_check_list;
            to_check_list.num_sol_n = 0;
            to_check_list.num_sol_p = 0;
            long SPAND = FINAL_STAGE_BLOCKSIZE/SECOND_STAGE_BLOCKSIZE;
            if (jnd > ind){
                for (long i = ind * SPAND; i < ind * SPAND + SPAND; i++){
                    for (long j = jnd * SPAND; j < jnd * SPAND + SPAND; j++){
                        to_check_list.num_sol_n += sol_table1[i][j-i].num_sol_n;
                        to_check_list.num_sol_p += sol_table1[i][j-i].num_sol_p;
                    }
                }
                to_check_list.solp = new long[to_check_list.num_sol_p];
                to_check_list.soln = new long[to_check_list.num_sol_n];
                long pind = 0;
                long nind = 0;
                for (long i = ind * SPAND; i < ind * SPAND + SPAND; i++){
                    for (long j = jnd * SPAND; j < jnd * SPAND + SPAND; j++){
                        for (long l = 0; l < sol_table1[i][j-i].num_sol_p; l++){
                            to_check_list.solp[pind] = sol_table1[i][j-i].solp[l];
                            pind++;
                        }
                        for (long l = 0; l < sol_table1[i][j-i].num_sol_n; l++){
                            to_check_list.soln[nind] = sol_table1[i][j-i].soln[l];
                            nind++;
                        }
                    }
                }
            }else{
                for (long i = ind * SPAND; i < ind * SPAND + SPAND; i++){
                    for (long j = i; j < jnd * SPAND + SPAND; j++){
                        to_check_list.num_sol_n += sol_table1[i][j-i].num_sol_n;
                        to_check_list.num_sol_p += sol_table1[i][j-i].num_sol_p;
                    }
                }
                to_check_list.solp = new long[to_check_list.num_sol_p];
                to_check_list.soln = new long[to_check_list.num_sol_n];
                long pind = 0;
                long nind = 0;
                for (long i = ind * SPAND; i < ind * SPAND + SPAND; i++){
                    for (long j = i; j < jnd * SPAND + SPAND; j++){
                        for (long l = 0; l < sol_table1[i][j-i].num_sol_p; l++){
                            to_check_list.solp[pind] = sol_table1[i][j-i].solp[l];
                            pind++;
                        }
                        for (long l = 0; l < sol_table1[i][j-i].num_sol_n; l++){
                            to_check_list.soln[nind] = sol_table1[i][j-i].soln[l];
                            nind++;
                        }
                    }
                }
            }
            set_zero(fp_buf[0], to_check_list.num_sol_p);
            set_zero(fp_buf[1], to_check_list.num_sol_n);
            for (long bias = 0; bias < num_dt[2]; bias += FINAL_STAGE_VECLENGTH){
                for (long l = 0; l < to_check_list.num_sol_p/2; l++){
                    float *src1 = dot_mat2[to_check_list.solp[l*2]]+bias;
                    float *src2 = dot_mat2[to_check_list.solp[l*2+1]]+bias;
                    fp_buf[0][l] += add_round_norm(src1, src2, FINAL_STAGE_VECLENGTH);
                }
                for (long l = 0; l < to_check_list.num_sol_n/2; l++){
                    float *src1 = dot_mat2[to_check_list.soln[l*2]]+bias;
                    float *src2 = dot_mat2[to_check_list.soln[l*2+1]]+bias;
                    fp_buf[1][l] += sub_round_norm(src1, src2, FINAL_STAGE_VECLENGTH);
                }
            }

            for (long l = 0; l < to_check_list.num_sol_p/2; l++){
                if (fp_buf[0][l] < dt2_bound){
                    buffer->solp[buffer->num_sol_p] = to_check_list.solp[l*2];
                    buffer->solp[buffer->num_sol_p+1] = to_check_list.solp[l*2+1];
                    buffer->num_sol_p+=2;
                }
            }
            for (long l = 0; l < to_check_list.num_sol_n/2; l++){
                if (fp_buf[1][l] < dt2_bound){
                    buffer->soln[buffer->num_sol_n] = to_check_list.soln[l*2];
                    buffer->soln[buffer->num_sol_n+1] = to_check_list.soln[l*2+1];
                    buffer->num_sol_n+=2;
                }
            }


            delete[] to_check_list.solp;
            delete[] to_check_list.soln;
        }
    }
    do {
        //free
        for (long i = 0; i < num_st/SECOND_STAGE_BLOCKSIZE; i++){
            for (long j = 0; j < num_st/SECOND_STAGE_BLOCKSIZE - i; j++){
                delete[] sol_table1[i][j].solp;
                delete[] sol_table1[i][j].soln;
            }
            delete[] sol_table1[i];
        }
        delete[] sol_table1;
        for (long i = 0; i < num_threads; i++){
            FREE_MAT(fp2_buf[i]);
        }
        delete[] fp2_buf;
        FREE_MAT(dot_mat1);
        FREE_MAT(dot_mat2);
    } while(0);
    gettimeofday(&dt2_end, NULL);
    double dt2_time = (dt2_end.tv_sec-dt2_start.tv_sec)+(double)(dt2_end.tv_usec-dt2_start.tv_usec)/1000000.0;
    long num_sol_stage2;
    if (show_details){
        long num_flops = num_sol_stage1*num_dt[2];
        num_sol_stage2 = 0;
        for (long i = 0; i < num_threads; i++){
            num_sol_stage2 += sol_list[i].num_sol_p;
            num_sol_stage2 += sol_list[i].num_sol_n;
        }
        num_sol_stage2 /= 2;
        //printf("done. time = %fs, found %ld pairs in total, speed = %f MFLOPS\n\n", dt2_time, num_sol_stage2, num_flops/(dt2_time*1000000));
    }

    // FINAL CHECK
    if (show_details) {
        printf("final check begin, %ld pairs will be checked by a partial sieving, goal = %.5f\n",
                num_sol_stage2, goal_length);
        std::cout << std::flush;
    }
    struct timeval fc_start, fc_end;
	gettimeofday(&fc_start, NULL);
    double **final_check_list = (double **) NEW_MAT(num_sol_stage2, d+dim_st, sizeof(double));
    double *success_len = (double *) NEW_VEC(num_sol_stage2, sizeof(double));
    long num_success = 0;
    do {
        long fcl_ind = 0;
        for (long i = 0; i < num_threads; i++){
            for (long j = 0; j < sol_list[i].num_sol_p; j+=2){
                double *src1 = st_vecs[sol_list[i].solp[j]];
                double *src2 = st_vecs[sol_list[i].solp[j+1]];
                add(final_check_list[fcl_ind], src1, src2, d+dim_st);
                fcl_ind++;
            }
            for (long j = 0; j < sol_list[i].num_sol_n; j+=2){
                double *src1 = st_vecs[sol_list[i].soln[j]];
                double *src2 = st_vecs[sol_list[i].soln[j+1]];
                sub(final_check_list[fcl_ind], src1, src2, d+dim_st);
                fcl_ind++;
            }
        }
    } while (0);
    //check by a partial sieve now
    for (long i = 0; i < num_sol_stage2; i++){
        Lattice_QP Lfc(d+1, d+1);
        double **bfc = Lfc.get_b().hi;
        double **b = Ld.get_b().hi;
        for (long j = 0; j < d; j++){
            copy(bfc[j], b[j], d);
        }
        for (long j = 0; j < d; j++) bfc[d][j] = final_check_list[i][j];
        bfc[d][d] = sqrt(Ld.get_B().hi[d-1])/10.0;
        Pool pool(&Lfc);
        pool.left_progressive_sieve(d+1-50, d+1, num_threads, 0);
        short *x = pool.min_lift_coeff(0);
        if (!x) continue;
        if (abs(x[d]) > 1.5) continue;
        double *vec_gso = (double *) NEW_VEC(d+ dim_st, sizeof(double));
        double **bw = Lw.get_b().hi;
        red(vec_gso, final_check_list[i], x[d], d+dim_st);
        for (long j = 0; j < d; j++){
            red(vec_gso, bw[j], x[j], d+dim_st);
        }
        if (show_details){
            double head = 0.0;
            double tail = 0.0;
            for (long j = 0; j < d; j++) head += vec_gso[j] * vec_gso[j];
            for (long j = d; j < d+dim_st; j++) tail += vec_gso[j] * vec_gso[j];
            printf("sol %ld, head = %.2f, tail = %.2f, norm = %.2f\n", i, sqrt(head), sqrt(tail), sqrt(head+tail));
        }
        if (dot(vec_gso, vec_gso, d+dim_st) > Ld.get_B().hi[0]) continue;
        double *vec_coeff = get_v_coeff(&Lw_dual, vec_gso); 
        bool fperr = false;
        for (long j = 0; j < d+dim_st; j++){
            if (abs(vec_coeff[j] - round(vec_coeff[j])) > 0.01){
                cout << "[Warning] final check: floating point error.\n";
                fperr = true;
                break;
            }
        }
        if (!fperr){
            double *vecb = (double *) NEW_VEC(L.NumCols(), sizeof(double));
            double **b = L.get_b().hi;
            for (long j = 0; j < d+dim_st; j++){
                red(vecb, b[j], round(vec_coeff[j]), L.NumCols());
            }
            if (sqrt(dot(vecb, vecb, L.NumCols())) < goal_length){
                success_len[num_success] = sqrt(dot(vecb, vecb, L.NumCols()));
                num_success++;
            }
            //cout << "length = " << sqrt(dot(vecb, vecb, L.NumCols())) << ", vec = \n";
            //print(vecb, L.NumCols());
            FREE_VEC(vecb);
        }
        FREE_VEC((void *)x);
        FREE_VEC(vec_gso);
        FREE_VEC(vec_coeff);
    }
    gettimeofday(&fc_end, NULL);
    double fc_time = (fc_end.tv_sec-fc_start.tv_sec)+(double)(fc_end.tv_usec-fc_start.tv_usec)/1000000.0;
    gettimeofday(&run_end, NULL);
    double run_time = (run_end.tv_sec-run_start.tv_sec)+(double)(run_end.tv_usec-run_start.tv_usec)/1000000.0;
    if (show_details){
        printf("final check done, time = %fs, total time = %fs, found %ld goal vecs, length = \n", fc_time, run_time, num_success);
        if (num_success > 0) {
            print(success_len, num_success);
        } else {
            cout << "[]\n";
        }
        
    }
    


    return 0;
}


/*  put buckets tegother
    long SPAND = SECOND_STAGE_BLOCKSIZE/FIRST_STAGE_BLOCKSIZE;
    PJT_DTG_sol_bucket to_check_list;
    to_check_list.num_sol_n = 0;
    to_check_list.num_sol_p = 0;
    if (jnd > ind){
        for (long i = ind * SPAND; i < ind * SPAND + SPAND; i++){
            for (long j = jnd * SPAND; j < jnd * SPAND + SPAND; j++){
                to_check_list.num_sol_n += sol_table0[i][j-i].num_sol_n;
                to_check_list.num_sol_p += sol_table0[i][j-i].num_sol_p;
            }
        }
        to_check_list.solp = new long[to_check_list.num_sol_p];
        to_check_list.soln = new long[to_check_list.num_sol_n];
        long pind = 0;
        long nind = 0;
        for (long i = ind * SPAND; i < ind * SPAND + SPAND; i++){
            for (long j = jnd * SPAND; j < jnd * SPAND + SPAND; j++){
                for (long i = )
                to_check_list.solp[pind] = sol_table0[i][j-i].solp;
                to_check_list.soln[nind] = sol_table0[i][j-i].num_sol_p;
            }
        }
    }else{

    }
*/
