
double shpc_maxabsdiff( int m, int n, double *A, int rsA, int csA, 
                                      double *B, int rsB, int csB );


int get_args( int argc, char **argv, int *nrepeats, int *first,                 
                                     int *last, int *inc );

void packing_a( double *A, double *packed_A, int m, int k, int csA, int rsA);

void packing_a( double *A, double *packed_A, int k, int n, int csB, int rsB);
