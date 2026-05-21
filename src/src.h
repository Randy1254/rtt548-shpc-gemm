
/* Computes C = AB + C */
void shpc_dgemm( int m, int n, int k, 
                    double *A, int rsA, int csA, 
                    double *B, int rsB, int csB, 
                    double *C, int rsC, int csC );

void shpc_ukernel_4x4( int k,                                                   
        double *A, int rsA, int csA,                                            
        double *B, int rsB, int csB,                                            
        double *C, int rsC, int csC );

void packing_a( double *A, double *packed_A, int m, int k, int csA, int rsA);

void packing_a( double *A, double *packed_A, int k, int n, int csB, int rsB);
                
