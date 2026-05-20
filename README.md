# High Performance GEMM 

## This project uses GOTO's Algorithm as a reference to implement an optimized function to perform GEMM or General Matrix by Matrix Multiplication by using techniques such as register and cache blocking, loop unrolling, and packing.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/BLIS_matrix-matrix_multiplication.png/960px-BLIS_matrix-matrix_multiplication.png?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail&_=20190401030024" height=240 width=160>

The idea of this algorithm is to efficiently use the different levels of caches to fit the matrices as during the operation there are many reuses of the different parts of the matrices. Since this is based on the size and associativity of the caches, each machine would have different cache blocking sizes.
