#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>

#include<immintrin.h>


#define dabs( x ) ( (x) < 0 ? -(x) : x )
#define max(x, y) (((x) > (y)) ? (x) : (y))


#include "blis.h"

#include "src.h"
#include "test.h"                                                        
#include "util.h"

#define MR 8
#define NR 6

#define MC 96
#define NC 7680
#define KC 256

#define min(x, y) ((x < y) ? x: y)
