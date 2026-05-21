#include "assignment3.h"


static double packed_B[KC * NC];
static double packed_A[MC * KC];

void shpc_dgemm( int m, int n, int k,                                            
        double *A, int rsA, int csA,                                
        double *B, int rsB, int csB,                                
        double *C, int rsC, int csC )
{

    // L3 Cache
    for(int j = 0; j < n; j += NC){

        // L2 Cache
        int remain_n = n - j;
        int min_nc = min(NC, remain_n);

        for(int p = 0; p < k; p += KC){

            int remain_k = k - p;
            int min_kc = min(KC, remain_k);

            // Pack B
            packing_b((double *)(B + j * csB + p * rsB), &packed_B, min_kc, min_nc, csB, rsB);
            

            // L1 Cache
            for(int i = 0; i < m; i += MC){

                int remain_m = m - i;
                int min_mc = min(MC, remain_m);
                

                packing_a((double *)(A + i * rsA + p * csA), &packed_A, min_mc, min_kc, csA, rsA);

                // Registers
                for(int jr = 0; jr < min_nc; jr += NR){

                    for(int ir = 0; ir < min_mc; ir += MR){

                        shpc_ukernel_8x6(min_kc, &packed_A[ir * min_kc], 1, MR, 
                                            &packed_B[min_kc * jr], NR, 1, 
                                            (double *)(C + rsC * ir + rsC * i + csC * jr + csC * j), rsC, csC);
                    }
                }

            }
        }
    }



}

void packing_a(double *A, double *packed_A, int m, int k, int csA, int rsA){


    for(int i = 0; i < m; i+=MR){

        int remain_ir = m - i;
        int min_ir = min(MR, remain_ir);

        for(int p = 0; p < k; p++){
            
            for(int ir = 0; ir < min_ir; ir++){
                packed_A[i * k + p * MR + ir] = *(A + i * rsA + p * csA + ir * rsA);
            }

            for(int ir = min_ir; ir < MR; ir++){
                packed_A[i * k + p * MR + ir] = 0;
            }
        }

    }
    
}

void packing_b(double *B, double *packed_B, int k, int n, int csB, int rsB){

    for(int j = 0; j < n; j += NR){

        for(int p = 0; p < k; p++){
            int remain_nr = n - j;
            int min_nr = min(NR, remain_nr);

            for(int jr = 0; jr < min_nr; jr++){
                packed_B[p * NR + j * k + jr] = *(B + j * csB + jr * csB + rsB * p);
            }


            for(int jr = min_nr; jr < NR; jr++){
                packed_B[p * NR + j * k + jr] = 0;
            }

        }
    }

}

void shpc_ukernel_8x6( int k,                                            
        double *A, int rsA, int csA,                                
        double *B, int rsB, int csB,                                
        double *C, int rsC, int csC )
{
    
    /* Declare vector registers to hold 4x4 C and load them */

    // Row 0-3
    __m256d gamma_0123_0 = _mm256_loadu_pd( &C[0*rsC + 0*csC] ); // Col 0 of C
    __m256d gamma_0123_1 = _mm256_loadu_pd( &C[0*rsC + 1*csC] ); // Col 1 of C
    __m256d gamma_0123_2 = _mm256_loadu_pd( &C[0*rsC + 2*csC] ); // Col 2 of C
    __m256d gamma_0123_3 = _mm256_loadu_pd( &C[0*rsC + 3*csC] ); // Col 3 of C
    __m256d gamma_0123_4 = _mm256_loadu_pd( &C[0*rsC + 4*csC] ); // Col 4 of C
    __m256d gamma_0123_5 = _mm256_loadu_pd( &C[0*rsC + 5*csC] ); // Col 5 of C

    // Row 4-7
    __m256d gamma_4567_0 = _mm256_loadu_pd( &C[4*rsC + 0*csC] ); // Col 0 of C
    __m256d gamma_4567_1 = _mm256_loadu_pd( &C[4*rsC + 1*csC] ); // Col 1 of C
    __m256d gamma_4567_2 = _mm256_loadu_pd( &C[4*rsC + 2*csC] ); // Col 2 of C
    __m256d gamma_4567_3 = _mm256_loadu_pd( &C[4*rsC + 3*csC] ); // Col 3 of C
    __m256d gamma_4567_4 = _mm256_loadu_pd( &C[4*rsC + 4*csC] ); // Col 4 of C
    __m256d gamma_4567_5 = _mm256_loadu_pd( &C[4*rsC + 5*csC] ); // Col 5 of C

    __m256d vector_array[] = {gamma_0123_0, gamma_0123_1, gamma_0123_2, gamma_0123_3, gamma_0123_4, gamma_0123_5,
                              gamma_4567_0, gamma_4567_1, gamma_4567_2, gamma_4567_3, gamma_4567_4, gamma_4567_5};

    for (int p=0; p<k; p++)
    {
        __m256d beta_p_j;

        // Col k, Row 0-3 of A
        __m256d alpha_0123_p = _mm256_loadu_pd( &A[ 0*rsA + p*csA ] );

        // Col k, Row 4-7
        __m256d alpha_4567_p = _mm256_loadu_pd( &A[ 4*rsA + p*csA ] );

        // Col 0 of C
        beta_p_j = _mm256_broadcast_sd( &B[ p*rsB + 0*csB]);
        vector_array[0] = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, vector_array[0]);
        vector_array[NR + 0] = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, vector_array[NR + 0]);

        // Col 1 of C
        beta_p_j = _mm256_broadcast_sd( &B[ p*rsB + 1*csB]);
        vector_array[1] = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, vector_array[1]);
        vector_array[NR + 1] = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, vector_array[NR + 1]);

        // Col 2 of C
        beta_p_j = _mm256_broadcast_sd( &B[ p*rsB + 2*csB]);
        vector_array[2] = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, vector_array[2]);
        vector_array[NR + 2] = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, vector_array[NR + 2]);

        // Col 3 of C
        beta_p_j = _mm256_broadcast_sd( &B[ p*rsB + 3*csB]);
        vector_array[3] = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, vector_array[3]);
        vector_array[NR + 3] = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, vector_array[NR + 3]);

        // Col 4 of C
        beta_p_j = _mm256_broadcast_sd( &B[ p*rsB + 4*csB]);
        vector_array[4] = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, vector_array[4]);
        vector_array[NR + 4] = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, vector_array[NR + 4]);

        // Col 5 of C
        beta_p_j = _mm256_broadcast_sd( &B[ p*rsB + 5*csB]);
        vector_array[5] = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, vector_array[5]);
        vector_array[NR + 5] = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, vector_array[NR + 5]);

    }

    
    for(int i = 0; i < NR; i++){
        _mm256_storeu_pd( &C[0*rsC + i*csC], vector_array[i] );
        _mm256_storeu_pd( &C[MR / 2 * rsC + i*csC], vector_array[NR + i]);
    }

}
   
